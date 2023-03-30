import fnmatch
import click
import collections
import mlflow
import json
import time
import os.path

from ast import literal_eval
from tempfile import TemporaryDirectory
from tqdm import tqdm

from aim import Run, Image, Text, Audio

IMAGE_EXTENSIONS = ('jpg', 'bmp', 'jpeg', 'png', 'gif', 'svg')
HTML_EXTENSIONS = ('html',)
TEXT_EXTENSIONS = (
    'txt',
    'log',
    'py',
    'js',
    'yaml',
    'yml',
    'json',
    'csv',
    'tsv',
    'md',
    'rst',
    'jsonnet',
)

# Audio is not handled in mlflow but including here just in case
AUDIO_EXTENSIONS = (
    'flac',
    'mp3',
    'wav',
)


class RunHashCache:
    def __init__(self, repo_path, no_cache=False):
        self._cache_path = os.path.join(repo_path, 'mlflow_logs_cache')
        self._needs_refresh = False

        if no_cache and os.path.exists(self._cache_path):
            os.remove(self._cache_path)

        try:
            with open(self._cache_path) as FS:
                self._cache = json.load(FS)
        except Exception:
            self._cache = {}

    def get(self, run_id):
        return self._cache.get(run_id)

    def __setitem__(self, key: str, val: str):
        if not self._cache.get(key) == val:
            self._cache[key] = val
            self._needs_refresh = True

    def __getitem__(self, key: str):
        return self._cache[key]

    def refresh(self):
        if self._needs_refresh:
            with open(self._cache_path, 'w') as FS:
                json.dump(self._cache, FS)


def get_mlflow_experiments(client, experiment):
    if experiment is None:
        # process all experiments
        experiments = client.search_experiments()
    else:
        try:
            ex = client.get_experiment(experiment)
        except mlflow.exceptions.MlflowException:
            ex = client.get_experiment_by_name(experiment)
        if not ex:
            raise RuntimeError(f'Could not find experiment with id or name "{experiment}"')
        experiments = (ex,)

    return experiments


def get_aim_run(repo_inst, run_id, run_name, experiment_name, run_cache):
    if run_cache.get(run_id):
        aim_run = Run(
            run_hash=run_cache[run_id],
            repo=repo_inst,
            system_tracking_interval=None,
            capture_terminal_logs=False,
            experiment=experiment_name,
        )
    else:
        aim_run = Run(
            repo=repo_inst,
            system_tracking_interval=None,
            capture_terminal_logs=False,
            experiment=experiment_name,
        )
        run_cache[run_id] = aim_run.hash
    aim_run.name = run_name
    return aim_run


def collect_run_params(aim_run, mlflow_run):
    aim_run['mlflow_run_id'] = mlflow_run.info.run_id
    aim_run['mlflow_experiment_id'] = mlflow_run.info.experiment_id
    aim_run.description = mlflow_run.data.tags.get("mlflow.note.content")

    # Collect params & tags
    # MLflow provides "string-ified" params values and we try to revert that
    aim_run['params'] = _map_nested_dicts(_try_parse_str, mlflow_run.data.params)
    aim_run['tags'] = {
        k: v for k, v in mlflow_run.data.tags.items() if not k.startswith('mlflow')
    }


def collect_artifacts(aim_run, mlflow_run, mlflow_client, exclude_artifacts):
    if '*' in exclude_artifacts:
        return

    run_id = mlflow_run.info.run_id

    artifacts_cache_key = '_mlflow_artifacts_cache'
    artifacts_cache = aim_run.meta_run_tree.get(artifacts_cache_key) or []

    __html_warning_issued = False
    with TemporaryDirectory(prefix=f'mlflow_{run_id}_') as temp_path:
        artifact_loc_stack = [None]
        while artifact_loc_stack:
            loc = artifact_loc_stack.pop()
            artifacts = mlflow_client.list_artifacts(run_id, path=loc)

            for file_info in artifacts:
                if file_info.is_dir:
                    artifact_loc_stack.append(file_info.path)
                    continue

                if file_info.path in artifacts_cache:
                    continue
                else:
                    artifacts_cache.append(file_info.path)

                if exclude_artifacts:
                    exclude = False
                    for expr in exclude_artifacts:
                        if fnmatch.fnmatch(file_info.path, expr):
                            exclude = True
                            break
                    if exclude:
                        continue

                downloaded_path = mlflow_client.download_artifacts(run_id, file_info.path, dst_path=temp_path)
                if file_info.path.endswith(HTML_EXTENSIONS):
                    if not __html_warning_issued:
                        click.secho(
                            'Handler for html file types is not yet implemented.', fg='yellow'
                        )
                        __html_warning_issued = True
                    continue
                elif file_info.path.endswith(IMAGE_EXTENSIONS):
                    aim_object = Image
                    kwargs = dict(
                        image=downloaded_path,
                        caption=file_info.path
                    )
                    content_type = 'image'
                elif file_info.path.endswith(TEXT_EXTENSIONS):
                    with open(downloaded_path) as fh:
                        content = fh.read()
                    aim_object = Text
                    kwargs = dict(
                        text=content
                    )
                    content_type = 'text'
                elif file_info.path.endswith(AUDIO_EXTENSIONS):
                    audio_format = os.path.splitext(file_info.path)[1].lstrip('.')
                    aim_object = Audio
                    kwargs = dict(
                        data=downloaded_path,
                        caption=file_info.path,
                        format=audio_format
                    )
                    content_type = 'audio'
                else:
                    click.secho(
                        f'Unresolved or unsupported type for artifact {file_info.path}', fg='yellow'
                    )
                    continue

                try:
                    item = aim_object(**kwargs)
                except Exception as exc:
                    click.echo(
                        f'Could not convert artifact {file_info.path} into aim object - {exc}', err=True
                    )
                    continue
                aim_run.track(item, name=loc or 'root', context={'type': content_type})

            aim_run.meta_run_tree[artifacts_cache_key] = artifacts_cache


def collect_metrics(aim_run, mlflow_run, mlflow_client, timestamp=None):
    for key in mlflow_run.data.metrics.keys():
        metric_history = mlflow_client.get_metric_history(mlflow_run.info.run_id, key)
        if timestamp:
            metric_history = list(filter(lambda m: m.timestamp >= timestamp, metric_history))

        for m in metric_history:
            aim_run.track(m.value, step=m.step, name=m.key)


def convert_existing_logs(repo_inst, tracking_uri, experiment=None, excluded_artifacts=None, no_cache=False):
    client = mlflow.tracking.client.MlflowClient(tracking_uri=tracking_uri)

    experiments = get_mlflow_experiments(client, experiment)
    run_cache = RunHashCache(repo_inst.path, no_cache)
    for ex in tqdm(experiments, desc=f'Parsing mlflow experiments in {tracking_uri}', total=len(experiments)):
        runs = client.search_runs(ex.experiment_id)

        for run in tqdm(runs, desc=f'Parsing mlflow runs for experiment `{ex.name}`', total=len(runs)):
            # get corresponding `aim.Run` object for mlflow run
            aim_run = get_aim_run(repo_inst, run.info.run_id, run.info.run_name, ex.name, run_cache)
            # Collect params and tags
            collect_run_params(aim_run, run)

            # Collect metrics
            collect_metrics(aim_run, run, client)

            # Collect artifacts
            collect_artifacts(aim_run, run, client, excluded_artifacts)

    run_cache.refresh()


def _wait_forever(watcher):
    try:
        while True:
            time.sleep(24 * 60 * 60)  # sleep for a day
    except KeyboardInterrupt:
        watcher.stop()


def _map_nested_dicts(fun, tree):
    if isinstance(tree, collections.abc.Mapping):
        return {k: _map_nested_dicts(fun, subtree) for k, subtree in tree.items()}
    else:
        return fun(tree)


def _try_parse_str(s):
    assert isinstance(s, str), f'Expected a string, got {s} of type {type(s)}'
    try:
        return literal_eval(s.strip())
    except:  # noqa: E722
        return s
