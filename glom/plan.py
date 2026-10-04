"""The glom "build phase": inspect specs before they execute.

glom specs are normally interpreted on the fly, so the same data
access written as a string (``'user.id'``), a :class:`~glom.Path`,
or a ``T`` expression (``T['user']['id']``) travels through
different code paths and, on failure, produces different error
messages. Branching specs such as :class:`~glom.Coalesce` and
:class:`~glom.Fill` make that even harder to follow, because a
missed branch can silently resolve to a default such as ``None``.

:func:`build` separates spec construction from execution. It walks a
spec once and produces a :class:`SpecPlan`: a tree of inspectable
nodes which lists every path the spec will access (via
:attr:`SpecPlan.paths`) and every branch it may take (via
:attr:`SpecPlan.branches`), each annotated with its *origin* -- the
dict key, container slot, or branch from which it came.

:meth:`SpecPlan.glom` executes the plan. On failure it raises a
:class:`PlanError` wrapping the original :class:`~glom.GlomError`,
annotated with the original path and the origin/branch that
produced it, so misspelled field names report identically no matter
how the access was written.
"""

from glom.core import (glom,
                       Path,
                       T,
                       TType,
                       Spec,
                       Coalesce,
                       Fill,
                       Pipe,
                       Auto,
                       GlomError,
                       PathAccessError,
                       CoalesceError,
                       bbrepr,
                       _MISSING)


_ROOT_ORIGIN = 'spec'
_ACCESS_OPS = ('.', 'P', '[')


class PathAccess:
    """A single path that a spec will access, together with its
    provenance.

    Attributes:
       path (Path): the normalized path. String specs, ``T``
         expressions made of plain attribute/item accesses, and
         :class:`~glom.Path` objects all normalize to the same
         :class:`~glom.Path`.
       origin (str): human-readable location of the access within the
         spec tree, e.g. ``"spec -> dict key 'user_id'"`` or
         ``"spec -> dict key 'tags' -> Coalesce branch 0"``.
       source: the original subspec object that produced this access
         (a string, :class:`~glom.Path`, or ``T`` expression).
    """
    def __init__(self, path, origin, source):
        self.path = path
        self.origin = origin
        self.source = source

    def __repr__(self):
        return f'{self.__class__.__name__}({self.path!r}, origin={self.origin!r})'

    def __eq__(self, other):
        if isinstance(other, PathAccess):
            return (self.path == other.path and self.origin == other.origin
                    and self.source is other.source)
        return NotImplemented

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result


class Branch:
    """One alternative of a branching spec (currently, a single
    subspec of a :class:`~glom.Coalesce`).

    Attributes:
       kind (str): the branching spec type, currently ``'coalesce'``.
       index (int): zero-based position of the branch.
       spec: the branch's subspec.
       plan (SpecPlan): the built plan for the branch's subspec.
    """
    def __init__(self, kind, index, spec, plan):
        self.kind = kind
        self.index = index
        self.spec = spec
        self.plan = plan

    @property
    def origin(self):
        return self.plan.origin

    @property
    def paths(self):
        return self.plan.paths

    def __repr__(self):
        return f'{self.__class__.__name__}({self.kind!r}, {self.index}, {bbrepr(self.spec)})'


class SpecPlan:
    """An inspectable, executable representation of a glom spec.

    Produced by :func:`build`. A node corresponds to one spec object;
    :attr:`accesses` are the paths accessed directly by that object,
    :attr:`children` are the plans of its subspecs (dict values,
    list/tuple entries, Coalesce branches, etc.), and
    :attr:`branch_points` lists any branches at this node.

    For :class:`~glom.Coalesce` nodes, :attr:`has_default` reports
    whether a fallback is configured. That makes the difference
    between "no value, matched a ``None`` default" and "match failed"
    visible before execution.
    """
    def __init__(self, spec, origin=_ROOT_ORIGIN, fill=False):
        self.spec = spec
        self.origin = origin
        self.fill = fill
        self.accesses = []
        self.children = []
        self.branch_points = []
        self.has_default = False
        self.default = _MISSING
        self._analyze()

    def _child(self, spec, step, fill=None):
        child_fill = self.fill if fill is None else fill
        return SpecPlan(spec, f'{self.origin} -> {step}', child_fill)

    def _analyze(self):
        spec = self.spec

        if not self.fill and isinstance(spec, str):
            self._record_path(Path.from_text(spec), spec)
        elif type(spec) is TType:
            path = _path_from_t(spec)
            if path is not None:
                self._record_path(path, spec)
        elif isinstance(spec, Path):
            self._record_path(spec, spec)
        elif isinstance(spec, Coalesce):
            if spec.default is not _MISSING:
                self.has_default = True
                self.default = spec.default
            elif spec.default_factory is not _MISSING:
                self.has_default = True
                self.default = spec.default_factory
            for i, subspec in enumerate(spec.subspecs):
                child = self._child(subspec, f'Coalesce branch {i}')
                self.children.append(child)
                self.branch_points.append(Branch('coalesce', i, subspec, child))
        elif isinstance(spec, Spec):
            self.children.append(self._child(spec.spec, 'Spec'))
        elif isinstance(spec, Fill):
            self.children.append(self._child(spec.spec, 'Fill', fill=True))
        elif isinstance(spec, Auto):
            self.children.append(self._child(spec.spec, 'Auto', fill=False))
        elif isinstance(spec, Pipe):
            for i, step in enumerate(spec.steps):
                self.children.append(self._child(step, f'Pipe step {i}'))
        elif isinstance(spec, dict):
            for key, subspec in spec.items():
                self.children.append(self._child(subspec, f'dict key {key!r}'))
        elif isinstance(spec, (list, tuple, set, frozenset)):
            type_name = type(spec).__name__
            for i, subspec in enumerate(spec):
                self.children.append(self._child(subspec, f'{type_name} item {i}'))
        # everything else (callables, literals, Val/Call/Invoke/...):
        # an opaque leaf which accesses no statically-visible paths.

    def _record_path(self, path, source):
        self.accesses.append(PathAccess(_canonical_path(path), self.origin, source))

    def _walk(self):
        yield self
        for child in self.children:
            for node in child._walk():
                yield node

    @property
    def paths(self):
        """All :class:`PathAccess` entries under this node, in spec order."""
        return [access for node in self._walk() for access in node.accesses]

    @property
    def branches(self):
        """All :class:`Branch` objects under this node, in spec order."""
        return [branch for node in self._walk() for branch in node.branch_points]

    def glom(self, target, **kw):
        """Execute the plan against *target*.

        On failure, raises :class:`PlanError` annotating the original
        :class:`~glom.GlomError` with the originating path and branch.
        """
        try:
            return glom(target, self.spec, **kw)
        except GlomError as e:
            raise self._annotate(e, target) from e

    def _annotate(self, exc, target):
        access = None
        coalesce_node = None
        if isinstance(exc, PathAccessError):
            access = _find_access(self, exc.path)
        elif isinstance(exc, CoalesceError):
            coalesce_node = _find_coalesce(self, exc.coal_obj)
            if coalesce_node is not None:
                access = _find_failed_branch_access(coalesce_node, exc)
        if access is None and coalesce_node is None:
            return exc
        return PlanError(exc, access=access, coalesce=coalesce_node,
                         target=target, spec=self.spec)

    def __repr__(self):
        return (f'{self.__class__.__name__}({bbrepr(self.spec)}, '
                f'paths={len(self.paths)}, branches={len(self.branches)})')


class PlanError(GlomError):
    """Error raised by :meth:`SpecPlan.glom`, wrapping the original
    :class:`~glom.GlomError` with build-phase provenance.

    Attributes:
       exc: the original error raised during execution.
       access (PathAccess): the accessed path which failed, if
         identifiable.
       coalesce (SpecPlan): the failing Coalesce node, if applicable.
       target: the target the plan was executed against.
       spec: the spec which produced the failing plan.
    """
    def __init__(self, exc, access=None, coalesce=None, target=None, spec=None):
        self.exc = exc
        self.access = access
        self.coalesce = coalesce
        self.target = target
        self.spec = spec

    @property
    def path(self):
        if self.access is not None:
            return self.access.path
        return getattr(self.exc, 'path', None)

    def get_message(self):
        context = []
        if self.access is not None:
            context.append(f'path {self.access.path!r} (from {self.access.origin})')
        if self.coalesce is not None:
            tried = len(self.coalesce.branch_points)
            if self.coalesce.has_default:
                outcome = f'{tried} branches missed, using configured default'
            else:
                outcome = f'all {tried} branches missed, no default configured'
            context.append(f'Coalesce at {self.coalesce.origin}: {outcome}')
        original_msg = self._original_message()
        lines = [self.__class__.__name__ + ': ' + '; '.join(context + [original_msg])]
        if self.target is not None:
            lines.append(' Target: ' + bbrepr(self.target))
        if self.spec is not None:
            lines.append(' Spec: ' + bbrepr(self.spec))
        return '\n'.join(lines)

    def _original_message(self):
        if isinstance(self.exc, PathAccessError):
            # render the underlying path in normalized form so that
            # string, Path, and T-spec accesses report identically
            try:
                canonical = _canonical_path(self.exc.path)
                path_part = canonical.values()[self.exc.part_idx]
            except (AttributeError, IndexError):
                canonical = self.exc.path
                path_part = canonical
            return ('could not access %r, part %r of %r, got error: %r'
                    % (path_part, self.exc.part_idx, canonical, self.exc.exc))
        get_message = getattr(self.exc, 'get_message', None)
        return get_message() if callable(get_message) else str(self.exc)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.exc!r})'


def build(spec):
    """Build-phase entry point: walk *spec* and return a
    :class:`SpecPlan` listing the paths it accesses and the branches
    it contains, without executing it.

    >>> plan = build({'user_id': 'user.id',
    ...               'tags': Coalesce('tags', default=[])})
    >>> [access.path.values() for access in plan.paths]
    [('user', 'id'), ('tags',)]
    >>> plan.paths[0].origin
    "spec -> dict key 'user_id'"
    >>> plan.paths[1].origin
    "spec -> dict key 'tags' -> Coalesce branch 0"
    """
    return SpecPlan(spec)


def _path_from_t(t):
    ops = t.__ops__
    if ops[0] is not T:
        return None
    for i in range(1, len(ops), 2):
        if ops[i] not in _ACCESS_OPS:
            return None
    return _canonical_path(Path(t))


def _canonical_path(path):
    # T['x'] records a '[' operation while Path('x') and the
    # 'x'-string shorthand record 'P'. Both are item accesses, so
    # normalize '[' to 'P' for inspection and error matching.
    ops = path.path_t.__ops__
    normalized = [ops[0]]
    for i, op in enumerate(ops[1:]):
        if i % 2 == 0 and op == '[':
            op = 'P'
        normalized.append(op)
    normalized_t = TType()
    normalized_t.__ops__ = tuple(normalized)
    return Path(normalized_t)


def _find_access(plan, error_path):
    if error_path is None:
        return None
    error_path = _canonical_path(error_path)
    best = None
    for access in plan.paths:
        try:
            matches = error_path.startswith(access.path)
        except (TypeError, ValueError):
            matches = error_path == access.path
        if matches and (best is None or len(access.path) > len(best.path)):
            best = access
    return best


def _find_coalesce(plan, coal_obj):
    for node in plan._walk():
        if node.spec is coal_obj:
            return node
    return None


def _find_failed_branch_access(coalesce_node, exc):
    # exc.skipped aligns with coalesce_node.branch_points: one entry
    # per tried subspec, in order. Report the first branch that failed
    # on a path access so the original misspelled path is visible.
    for branch, skipped in zip(coalesce_node.branch_points, exc.skipped):
        if isinstance(skipped, PathAccessError):
            access = _find_access(branch.plan, skipped.path)
            if access is not None:
                return access
    return None
