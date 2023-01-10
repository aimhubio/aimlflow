import click
import os


from click import core
from click import ClickException

from aim.sdk.repo import Repo
from aim.sdk.utils import clean_repo_path

from aimlflow.utils import convert_existing_logs
from aimlflow.watcher import MLFlowWatcher
core._verify_python3_env = lambda: None


@click.group()
@click.option('-v', '--verbose', is_flag=True)
def cli_entry_point(verbose):
    if verbose:
        click.echo('Verbose mode is on')


@cli_entry_point.command(name='convert')
@click.option('--aim-repo', required=False, type=click.Path(exists=True,
                                                            file_okay=False,
                                                            dir_okay=True,
                                                            writable=True))
@click.option('--mlflow-tracking-uri',  required=False, default=None)
@click.option('--experiment', '-e', required=False, default=None)
@click.option('--watch', required=False, is_flag=True, default=False)
def convert(aim_repo, mlflow_tracking_uri, experiment, watch):
    repo_path = clean_repo_path(aim_repo) or Repo.default_repo_path()
    repo_inst = Repo.from_path(repo_path)

    mlflow_tracking_uri = mlflow_tracking_uri or os.environ.get('MLFLOW_TRACKING_URI')
    if not mlflow_tracking_uri:
        raise ClickException('MLFlow tracking URI must be provided either through ENV or CLI.')

    if watch:
        watcher = MLFlowWatcher(repo_inst, mlflow_tracking_uri, experiment)

    click.echo('Converting existing MLflow logs.')
    convert_existing_logs(repo_inst, mlflow_tracking_uri, experiment)

    if watch:
        click.echo(f'Starting watcher on {mlflow_tracking_uri}.')
        watcher.start()
        from aimlflow.utils import _wait_forever
        _wait_forever(watcher)
