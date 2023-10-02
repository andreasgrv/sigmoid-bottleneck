import os


class MLBLPath(object):

    BP_FILE = 'blueprint.yaml'
    MODEL_FILE = 'model'
    STATS_FILE = 'stats.tsv'
    OUT_FOLDER = 'output'
    CHECKPOINT_FOLDER = 'checkpoints'
    ANALYSIS_FOLDER = 'analysis'

    # We set some basic paths in the following environment variables
    MLBL_PATH = 'MLBL_PATH'
    EXP_PATH = 'experiments'

    """Utility class that provides easy access to common paths."""
    def __init__(self, experiment_name='default', root_folder=None):
        super(MLBLPath, self).__init__()
        self.experiment_name = experiment_name
        self.root_folder = root_folder or os.environ[self.MLBL_PATH]

        if not os.path.isdir(self.experiment_folder):
            print('Creating directory %s' % self.experiment_folder)
            os.makedirs(self.experiment_folder)
        if not os.path.isdir(self.checkpoint_folder):
            os.makedirs(self.checkpoint_folder)
        if not os.path.isdir(self.analysis_folder):
            os.makedirs(self.analysis_folder)

    @property
    def experiment_folder(self):
        return os.path.join(self.experiments_folder, self.experiment_name)

    @property
    def experiments_folder(self):
        return os.path.join(self.root_folder, self.EXP_PATH)

    @property
    def vocab_folder(self):
        return self.experiment_folder

    @property
    def output_folder(self):
        return os.path.join(self.experiment_folder, MLBLPath.OUT_FOLDER)

    @property
    def checkpoint_folder(self):
        return os.path.join(self.experiment_folder, MLBLPath.CHECKPOINT_FOLDER)

    @property
    def analysis_folder(self):
        return os.path.join(self.experiment_folder, MLBLPath.ANALYSIS_FOLDER)

    def analysis_for(self, split):
        return os.path.join(self.analysis_folder, '%s.json' % split)

    @property
    def model(self):
        return os.path.join(self.experiment_folder, '%s.pt' % MLBLPath.MODEL_FILE)

    def model_at_step(self, step):
        return os.path.join(self.checkpoint_folder, '%s@%d.pt' % (MLBLPath.MODEL_FILE, step))

    @property
    def stats(self):
        return os.path.join(self.experiment_folder, MLBLPath.STATS_FILE)

    @property
    def blueprint(self):
        return os.path.join(self.experiment_folder, MLBLPath.BP_FILE)

    def for_output(self, filename):
        file_path = os.path.join(self.output_folder, filename)
        return file_path
