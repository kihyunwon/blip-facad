import h5py
import datasets


class HDF5Config(datasets.BuilderConfig):

    def __init__(self, key='', **kwargs):
        """BuilderConfig for HDF5 file.
        """
        # Version history:
        # 0.0.1: Initial version.
        super(HDF5Config, self).__init__(version=datasets.Version("0.0.1"), **kwargs)
        self.key = key


class HDF5(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        HDF5Config(
            name="keyed_config",
            description="HDF5 Dataset Generator iterates values of provided key",
            key=''
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(description=self.config.description)

    def _split_generators(self, dl_manager):
        if not self.config.data_files:
            raise ValueError(f"At least one data file must be specified, but got data_files={self.config.data_files}")
        dl_manager.download_config.extract_on_the_fly = True
        data_files = dl_manager.download_and_extract(self.config.data_files)
        splits = []
        for split_name, files in data_files.items():
            splits.append(datasets.SplitGenerator(name=split_name, gen_kwargs={"files": files}))
        return splits

    def _generate_examples(self, files):
        key = self.config.key
        for file in files:
            with h5py.File(file, "r", swmr=True) as data:
                if not key:
                    raise ValueError(f"A key must be specified, but got key={key}")
                else:
                    for idx, value in enumerate(data[key]):
                        yield idx, { key: value }
