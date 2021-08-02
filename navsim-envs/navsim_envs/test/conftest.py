# def pytest_addoption(parser):
#    parser.addoption("--env_file", action="store", default=None)

import pytest

import navsim_envs

def pytest_report_teststatus(report):
    category, short, verbose = '', '', ''
    if hasattr(report, 'wasxfail'):
        if report.skipped:
            category = 'xfailed'
            verbose = 'xfail'
        elif report.passed:
            category = 'xpassed'
            verbose = ('XPASS', {'yellow': True})
        return category, verbose + '\n', verbose
    elif report.when in ('setup', 'teardown'):
        if report.failed:
            category = 'error'
            verbose = 'ERROR'
        elif report.skipped:
            category = 'skipped'
            verbose = 'SKIPPED'
        return category, verbose + '\n', verbose
    category = report.outcome
    verbose = category.upper()
    return category, verbose + '\n', verbose


@pytest.fixture(scope="session",autouse=True)
def session_begin():
    print(f'{navsim_envs.__version_banner__}')
