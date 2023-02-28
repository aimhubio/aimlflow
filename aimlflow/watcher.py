import logging
import time

from threading import Thread
from typing import Union, TYPE_CHECKING, Dict

import mlflow.entities
from mlflow import MlflowClient
from aimlflow.utils import (
    get_mlflow_experiments,
    get_aim_run,
    collect_metrics,
    collect_artifacts,
    collect_run_params,
    RunHashCache
)

if TYPE_CHECKING:
    from aim import Repo, Run

logger = logging.getLogger(__name__)


class MLFlowWatcher:
    WATCH_INTERVAL_DEFAULT = 10

    def __init__(self,
                 repo: 'Repo',
                 tracking_uri: str,
                 experiment: str = None,
                 exclude_artifacts: str = None,
                 interval: Union[int, float] = WATCH_INTERVAL_DEFAULT,
                 ):

        self._last_watch_time = time.time()
        self._active_aim_runs_pool: Dict[str, 'Run'] = dict()

        self._watch_interval = interval

        self._client = MlflowClient(tracking_uri)

        self._exclude_artifacts = exclude_artifacts
        self._experiment = experiment
        self._experiments = get_mlflow_experiments(self._client, self._experiment)
        self._repo = repo

        self._th_collector = Thread(target=self._watch, daemon=True)
        self._shutdown = False
        self._started = False

    def start(self):
        if self._started:
            return

        self._started = True
        self._th_collector.start()

    def stop(self):
        if not self._started:
            return

        self._shutdown = True
        self._th_collector.join()

    def _search_experiment(self, experiment_id):
        return next((exp for exp in self._experiments if exp.experiment_id == experiment_id), None)

    def _get_current_active_mlflow_runs(self):
        experiment_ids = [ex.experiment_id for ex in self._experiments]

        active_runs = self._client.search_runs(
            experiment_ids=experiment_ids,
            run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
            filter_string='attribute.status="RUNNING"'
        )

        return active_runs

    def _process_single_run(self, aim_run, mlflow_run):
        # Collect params and tags
        collect_run_params(aim_run, mlflow_run)

        # Collect metrics
        collect_metrics(aim_run, mlflow_run, self._client, timestamp=self._last_watch_time)

        # Collect artifacts
        collect_artifacts(aim_run, mlflow_run, self._client, self._exclude_artifacts)

    def _process_runs(self):
        watch_started_time = time.time()

        # refresh experiments list
        self._experiments = get_mlflow_experiments(self._client, self._experiment)

        # process active runs
        active_mlflow_runs = self._get_current_active_mlflow_runs()

        run_cache = RunHashCache(self._repo.path)
        active_mlflow_run_ids = set()

        for mlflow_run in active_mlflow_runs:
            mlflow_run_id = mlflow_run.info.run_id
            active_mlflow_run_ids.add(mlflow_run_id)
            mlflow_experiment = self._search_experiment(mlflow_run.info.experiment_id)
            if self._active_aim_runs_pool.get(mlflow_run_id):
                aim_run = self._active_aim_runs_pool[mlflow_run_id]
            else:
                aim_run = get_aim_run(self._repo,
                                      mlflow_run.info.run_id,
                                      mlflow_run.info.run_name,
                                      mlflow_experiment.name,
                                      run_cache)
                self._active_aim_runs_pool[mlflow_run_id] = aim_run

            self._process_single_run(aim_run, mlflow_run)

        # process closed runs
        all_mlflow_run_ids = set(self._active_aim_runs_pool.keys())
        closed_mlflow_run_ids = all_mlflow_run_ids.difference(active_mlflow_run_ids)

        for mlflow_run_id in closed_mlflow_run_ids:
            # process closed run and remove from pool
            mlflow_run = self._client.get_run(mlflow_run_id)
            aim_run = self._active_aim_runs_pool[mlflow_run_id]
            self._process_single_run(aim_run, mlflow_run)
            aim_run.close()
            del self._active_aim_runs_pool[mlflow_run_id]

        # refresh runs cache and update timer
        self._last_watch_time = watch_started_time
        run_cache.refresh()

    def _watch(self):
        self._process_runs()
        watch_interval_counter = 0
        while True:
            if self._shutdown:
                break

            time.sleep(1)
            watch_interval_counter += 1

            if watch_interval_counter > self._watch_interval:
                self._process_runs()
                watch_interval_counter = 0
