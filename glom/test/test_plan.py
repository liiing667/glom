"""Tests for the spec build phase (glom.plan): build() produces an
inspectable SpecPlan, and SpecPlan.glom() executes it with errors
annotated by original path and branch."""

import pytest

from glom import (glom, build, Spec, Coalesce, Fill, Pipe, Path, T, S, Val,
                  GlomError, PathAccessError, CoalesceError, PlanError,
                  SpecPlan, PathAccess, Branch)


def test_build_lists_paths_with_origins():
    spec = Spec({'user_id': 'user.id', 'tags': Coalesce('tags', default=[])})
    plan = spec.build()

    assert isinstance(plan, SpecPlan)
    assert len(plan.paths) == 2

    user_access, tags_access = plan.paths
    assert isinstance(user_access, PathAccess)
    assert user_access.path == Path('user', 'id')
    assert "dict key 'user_id'" in user_access.origin

    assert tags_access.path == Path('tags')
    assert "dict key 'tags'" in tags_access.origin
    assert 'Coalesce branch 0' in tags_access.origin


def test_build_normalizes_path_t_and_string_specs():
    # the same access written three ways produces the same path
    plans = [build('a.b'), build(T['a']['b']), build(Path('a', 'b')), build(T.a.b)]
    paths = [plan.paths[0].path for plan in plans]
    assert paths[0] == paths[1] == paths[2]
    assert paths[3].values() == ('a', 'b')

    # and each access remembers how it was written
    assert plans[0].paths[0].source == 'a.b'
    assert plans[1].paths[0].source == T['a']['b'] or plans[1].paths[0].source is not None
    assert plans[2].paths[0].source == Path('a', 'b')


def test_build_lists_coalesce_branches():
    plan = build({'tags': Coalesce('tags', 'tag_list', default=[])})

    branches = plan.branches
    assert len(branches) == 2
    assert all(isinstance(branch, Branch) for branch in branches)
    assert [branch.kind for branch in branches] == ['coalesce', 'coalesce']
    assert [branch.index for branch in branches] == [0, 1]
    assert [branch.spec for branch in branches] == ['tags', 'tag_list']

    # each branch exposes its own paths
    assert branches[0].paths[0].path == Path('tags')
    assert branches[1].paths[0].path == Path('tag_list')

    # the coalesce node advertises its default
    coalesce_node = plan.children[0]
    assert coalesce_node.has_default
    assert coalesce_node.default == []


def test_build_distinguishes_missing_default_from_none_default():
    # a Coalesce without a default raises on miss; one with
    # default=None silently returns None. The plan makes the
    # difference visible before execution.
    no_default = build({'v': Coalesce('a', 'b')}).children[0]
    assert not no_default.has_default

    none_default = build({'v': Coalesce('a', 'b', default=None)}).children[0]
    assert none_default.has_default
    assert none_default.default is None

    factory_default = build({'v': Coalesce('a', default_factory=list)}).children[0]
    assert factory_default.has_default
    assert factory_default.default is list


def test_build_nested_coalesce_and_fill():
    spec = {'outer': Coalesce(Fill((T['a'], 'literal')), 'fallback')}
    plan = build(spec)

    # T['a'] inside Fill is a path; the string 'literal' inside Fill
    # is a literal value, not a path
    paths = plan.paths
    assert len(paths) == 2
    assert paths[0].path == Path('a')
    assert 'Fill' in paths[0].origin
    assert 'Coalesce branch 0' in paths[0].origin
    assert paths[1].path == Path('fallback')
    assert 'Coalesce branch 1' in paths[1].origin

    assert len(plan.branches) == 2


def test_build_containers_and_pipe():
    plan = build({'a': ['x', ('y', len)], 'b': Pipe('p.q', 'r')})
    paths = {access.path.values(): access.origin for access in plan.paths}
    assert ('x',) in paths and 'list item 0' in paths[('x',)]
    assert ('y',) in paths and 'tuple item 0' in paths[('y',)]
    assert ('p', 'q') in paths and 'Pipe step 0' in paths[('p', 'q')]
    assert ('r',) in paths and 'Pipe step 1' in paths[('r',)]


def test_build_skips_non_path_specs():
    # T expressions with calls, scope accesses, callables, and
    # literals access no statically-visible target paths
    assert build(T['a'].bit_length()).paths == []
    assert build(S['var']).paths == []
    assert build(len).paths == []
    assert build(Val('a.b')).paths == []
    assert build(None).paths == []


def test_plan_glom_matches_glom():
    target = {'user': {'id': 7}, 'tags': ['x']}
    spec = {'user_id': 'user.id', 'tags': Coalesce('tags', default=[])}
    plan = build(spec)
    assert plan.glom(target) == glom(target, spec)

    assert build(Coalesce('tags', default=[])).glom({}) == []

    missing_spec = {'user_id': 'user.id'}
    with pytest.raises(PathAccessError):
        glom({}, missing_spec)
    with pytest.raises(PlanError):
        build(missing_spec).glom({})


def test_plan_error_reports_original_path_and_origin():
    plan = build({'user_id': 'user.id'})
    with pytest.raises(PlanError) as exc_info:
        plan.glom({'user': {}})

    err = exc_info.value
    assert isinstance(err, GlomError)
    assert isinstance(err.exc, PathAccessError)
    assert err.access.path == Path('user', 'id')
    assert err.path == Path('user', 'id')
    message = err.get_message()
    assert "dict key 'user_id'" in message
    assert "Path('user', 'id')" in message


def test_plan_error_same_message_for_path_t_and_string():
    # a misspelled field reports the same path and origin regardless
    # of how the access was written
    messages = []
    for subspec in ('user.id', T['user']['id'], Path('user', 'id')):
        plan = build({'user_id': subspec})
        with pytest.raises(PlanError) as exc_info:
            plan.glom({'user': {}})
        message = exc_info.value.get_message()
        assert "dict key 'user_id'" in message
        assert "Path('user', 'id')" in message
        messages.append(message)
    # the provenance line is identical; only the Spec: line differs
    first_lines = [message.splitlines()[0] for message in messages]
    assert first_lines[0] == first_lines[1] == first_lines[2]


def test_plan_error_reports_coalesce_branch():
    plan = build({'tags': Coalesce('tags', 'tag_list')})
    with pytest.raises(PlanError) as exc_info:
        plan.glom({})

    err = exc_info.value
    assert isinstance(err.exc, CoalesceError)
    assert err.coalesce is not None
    message = err.get_message()
    assert "dict key 'tags'" in message
    assert 'Coalesce branch 0' in message
    assert 'all 2 branches missed' in message


def test_plan_error_unmatched_passes_through():
    # errors with no identifiable path or branch are re-raised as-is
    plan = build(lambda target: 1 / 0)
    with pytest.raises(GlomError) as exc_info:
        plan.glom({})
    assert not isinstance(exc_info.value, PlanError)


def test_plan_repr():
    plan = build({'a': Coalesce('b', default=1)})
    assert 'SpecPlan' in repr(plan)
    assert 'paths=1' in repr(plan)
    assert 'branches=1' in repr(plan)
    assert 'PathAccess' in repr(plan.paths[0])
    assert 'coalesce' in repr(plan.branches[0])
