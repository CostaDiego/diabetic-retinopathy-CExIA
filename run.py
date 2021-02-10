import luigi
from source import model, preProcess
from os import path

TASK_FOLDER = 'taskFile'

class PreProcess(luigi.Task):
    data_folder = luigi.Parameter()

    def requires(self):
        return None

    def output(self):
        return luigi.LocalTarget(
            path.join(
                TASK_FOLDER,
                '{file}.txt'.format(file = self.__class__.__name__)
            )
        )

    def run(self):
        print('Pre Process Task')
        print(self.data_folder+' --00--')

        with self.output().open('w') as out_file:
            print(self.__class__.__name__, file=out_file)

class Train(luigi.Task):
    data_folder = luigi.Parameter()

    def requires(self):
        return PreProcess(self.data_folder)

    def output(self):
        return luigi.LocalTarget(
            path.join(
                TASK_FOLDER,
                '{file}.txt'.format(file = self.__class__.__name__)
            )
        )

    def run(self):
        print('Train Task')
        print(self.data_folder+' --01--')

        with self.output().open('w') as out_file:
            print(self.__class__.__name__, file=out_file)


if __name__ == '__main__':
    pass