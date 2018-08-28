'''
Created on Aug 27, 2018

@author: Clark
'''

# ----- Insert that snippet to run distributed jobs -----

import os
import tensorflow as tf
from clusterone import get_data_path, get_logs_path

DATASET_NAME = ""
LOCAL_REPO = ""

class distributed_env():
    '''
    Specifying paths when working locally
    For convenience we use a clusterone wrapper (get_data_path below) to be able
    to switch from local to clusterone without cahnging the code.
    '''

    def __init__(self, local_data_path=None,
                 cloud_data_path=None,
                 logs_path=None,
                 local_repo=None,
                 cloud_user_repo=None,
                 flags=tf.app.flags.FLAGS):

        self.data_path = local_data_path
        self.logs_path = logs_path
        self.local_repo = local_repo
        self.cloud_data_path = cloud_data_path
        self.flags = flags
        return self.flags

    def get_env(self, data_subfolder=None):
        # Configure  distributed task
        try:
            job_name = os.environ['JOB_NAME']
            task_index = os.environ['TASK_INDEX']
            ps_hosts = os.environ['PS_HOSTS']
            worker_hosts = os.environ['WORKER_HOSTS']
        except:
            job_name = None
            task_index = 0
            ps_hosts = None
            worker_hosts = None

        flags = self.flags
        # Flags for configuring the distributed task
        flags.DEFINE_string("job_name", job_name,
                            "job name: worker or ps")

        flags.DEFINE_integer("task_index", task_index,
                             "Worker task index, should be >= 0. task_index=0 is "
                             "the chief worker task that performs the variable "
                             "initialization and checkpoint handling")
        flags.DEFINE_string("ps_hosts", ps_hosts,
                            "Comma-separated list of hostname:port pairs")
        flags.DEFINE_string("worker_hosts", worker_hosts,
                            "Comma-separated list of hostname:port pairs")

        # Training related flags
        flags.DEFINE_string("data_dir",
                            get_data_path(
                                dataset_name = self.cloud_user_repo, #all mounted repo
                                local_root = self.data_path,
                                local_repo = self.local_repo,
                                path = self.cloud_data_path
                                ),
                            "Path to dataset. It is recommended to use get_data_path()"
                            "to define your data directory.so that you can switch "
                            "from local to clusterone without changing your code."
                            "If you set the data directory manually make sure to use"
                            "/data/ as root path when running on ClusterOne cloud.")

        flags.DEFINE_string("log_dir",
                             get_logs_path(root=self.logs_path),
                            "Path to store logs and checkpoints. It is recommended"
                            "to use get_logs_path() to define your logs directory."
                            "so that you can switch from local to clusterone without"
                            "changing your code."
                            "If you set your logs directory manually make sure"
                            "to use /logs/ when running on ClusterOne cloud.")

    def device_and_target(self):
        # If FLAGS.job_name is not set, we're running single-machine TensorFlow.
        # Don't set a device.
        flags = self.flags
        if flags.job_name is None:
            print("Running single-machine training")
            return (None, "")

        # Otherwise we're running distributed TensorFlow
        print("Running distributed training")
        if flags.task_index is None or flags.task_index == "":
            raise ValueError("Must specify an explicit `task_index`")

        if flags.ps_hosts is None or flags.ps_hosts == "":
            raise ValueError("Must specify an explicit `ps_hosts`")

        if flags.worker_hosts is None or flags.worker_hosts == "":
            raise ValueError("Must specify an explicit `worker_hosts`")

        cluster_spec = tf.train.ClusterSpec({
              "ps": flags.ps_hosts.split(","),
              "worker": flags.worker_hosts.split(","),
          })

        server = tf.train.Server(
            cluster_spec, job_name=flags.job_name, task_index=flags.task_index)
        if flags.job_name == "ps":
            server.join()

        worker_device = "/job:worker/task:{}".format(flags.task_index)

        # The device setter will automatically place Variables ops on separate
        # parameter servers (ps). The non-Variable ops will be placed on the workers.
        return (
                tf.train.replica_device_setter(
                    worker_device=worker_device,
                    cluster=cluster_spec),
                server.target,
                )

# --- end of snippet ----
