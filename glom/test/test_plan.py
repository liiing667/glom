import pytest

from glom import (glom, Spec, SpecPlan, PlanNode, StructNode, PathNode,
                  BranchNode, LeafNode, Path, T, S, Coalesce, Fill, Val,
                  PathAccessError, CoalesceError)


def test_build_basic_paths_and_branches():
    plan = Spec({'user_id': 'user.id',
                 'tags': Coalesce('tags', default=[])}).build()

    assert isinstance(plan, SpecPlan)
    assert plan.paths == (Path('user', 'id'), Path('tags'))

    assert len(plan.branches) == 1
    branch = plan.branches[0]
    assert isinstance(branch, BranchNode)
    assert branch.spec.subspecs == ('tags',)
    assert len(branch.options) == 1
    assert isinstance(branch.options[0], PathNode)
    assert branch.options[0].path == Path('tags')


def test_build_path_spelling_equivalence():
    # str, Path, and T spellings of the same access build to the same path
    for spec in ('user.id', Path('user', 'id'), T['user']['id'], T.user.id):
        plan = Spec(spec).build()
        assert plan.paths == (Path('user', 'id'),)
        assert isinstance(plan.root, PathNode)


def test_build_nested_structures():
    plan = Spec({'a': ['x', 'y'], 'b': ('c', {'d': 'e.f'})}).build()
    assert plan.paths == (Path('x'), Path('y'), Path('c'), Path('e', 'f'))
    assert isinstance(plan.root, StructNode)
    assert plan.root.keys == ('a', 'b')


def test_build_dedupes_paths():
    plan = Spec({'a': 'x', 'b': 'x'}).build()
    assert plan.paths == (Path('x'),)


def test_build_fill_mode_literals():
    # inside Fill, strings are literals, not paths; the plan makes
    # this visible instead of silently returning the literal at runtime
    plan = Spec(Fill(('a', T['b']))).build()
    assert plan.paths == (Path('b'),)
    assert isinstance(plan.root.children[0], LeafNode)
    assert plan.root.children[0].role == 'literal'


def test_build_coalesce_under_fill_has_no_paths():
    # the classic silent-None trap: Coalesce subspecs under Fill are
    # literals, so no paths are accessed and defaults always win
    plan = Spec(Fill((Coalesce('a', 'b'),))).build()
    assert plan.paths == ()
    assert len(plan.branches) == 1
    assert all(isinstance(opt, LeafNode) for opt in plan.branches[0].options)


def test_build_val_and_callable_and_opaque():
    plan = Spec({'v': Val('a.b'), 'n': len, 't': T['x'] + 1}).build()
    assert plan.paths == ()
    roles = [child.role for child in plan.root.children]
    assert roles == ['literal', 'call', 'opaque']


def test_build_unwraps_spec_and_auto():
    from glom import Auto
    plan = Spec(Spec({'a': Auto(Fill(('x', 'y.z')))})).build()
    assert plan.paths == ()
    plan2 = Spec(Auto(Fill((T['q'],)))).build()
    assert plan2.paths == (Path('q'),)


def test_plan_glom_matches_spec_glom():
    spec = Spec({'user_id': 'user.id', 'tags': Coalesce('tags', default=[])})
    plan = spec.build()
    target = {'user': {'id': 1}}
    expected = {'user_id': 1, 'tags': []}
    assert plan.glom(target) == spec.glom(target) == expected

    target2 = {'user': {'id': 2}, 'tags': ['a']}
    assert plan.glom(target2) == {'user_id': 2, 'tags': ['a']}


def test_plan_glom_kwargs_passthrough():
    plan = Spec('a.b').build()
    assert plan.glom({}, default='dflt') == 'dflt'
    assert plan.glom({'a': {'b': 1}}) == 1


def test_plan_glom_scope():
    plan = Spec({'c': S['cat']}, scope={'cat': 1}).build()
    assert plan.glom(5) == {'c': 1}
    # call-time scope overrides
    assert plan.glom(5, scope={'cat': 2}) == {'c': 2}


def test_plan_glom_normalizes_error_paths():
    target = {'user': {'name': 'n'}}
    errors = []
    for spec in ('user.id', Path('user', 'id'), T['user']['id'], T.user.id):
        with pytest.raises(PathAccessError) as exc_info:
            Spec(spec).build().glom(target)
        errors.append(exc_info.value)

    # all spellings report the same canonical Path
    for err in errors:
        assert err.path == Path('user', 'id')
        assert repr(err.path) == "Path('user', 'id')"

    # getitem-based spellings produce byte-identical messages
    getitem_msgs = [err.get_message() for err in errors[:3]]
    assert getitem_msgs[0] == getitem_msgs[1] == getitem_msgs[2]

    # attribute access still reports its own operation, but on the
    # same canonical path
    assert "part 0 of Path('user', 'id')" in errors[3].get_message()


def test_plan_glom_error_normalization_is_local():
    # plain glom() behavior is untouched by plan normalization
    target = {'user': {'name': 'n'}}
    with pytest.raises(PathAccessError) as exc_info:
        glom(target, T.user.id)
    assert repr(exc_info.value.path) == 'T.user.id'


def test_plan_coalesce_error_carries_normalized_branches():
    plan = Spec(Coalesce('a', T['b'])).build()
    assert len(plan.branches) == 1

    with pytest.raises(CoalesceError) as exc_info:
        plan.glom({})
    skipped = exc_info.value.skipped
    assert len(skipped) == 2
    assert all(isinstance(s, PathAccessError) for s in skipped)
    assert [s.path for s in skipped] == [Path('a'), Path('b')]


def test_plan_describe_and_repr():
    plan = Spec({'user_id': 'user.id',
                 'tags': Coalesce('tags', default=[])}).build()
    text = plan.describe()
    assert "'user_id':" in text
    assert "path: Path('user', 'id')" in text
    assert 'branch:' in text
    assert "path: Path('tags')" in text

    assert repr(plan).startswith('SpecPlan({')
    assert repr(plan.branches[0]).startswith('BranchNode(Coalesce(')
    assert repr(plan.root.children[0]) == "PathNode(Path('user', 'id'))"


def test_plan_node_base_aggregate():
    node = PlanNode('whatever', children=[
        PathNode('a', Path('a')),
        BranchNode(Coalesce('b'), [PathNode('b', Path('b'))]),
    ])
    assert node.paths == (Path('a'), Path('b'))
    assert len(node.branches) == 1
    assert node.describe().splitlines()[0] == "node: 'whatever'"
