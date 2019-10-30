import pickle


class IOTools:
    """
    This class implements the persistence mechanism.
    """

    @staticmethod
    def save_to_file(data, file):
        """
        Saves selected variables to file.
        :param data: the selected variables to save
        :param file: file name and path
        :return:
        """

        with open(file, 'w') as f:
            pickle.dump(data, f)

    @staticmethod
    def load_from_file(file):
        """
        Loads selected variables from file.
        :param file: the file to be loaded
        :return: the loaded data
        """

        with open(file, 'r') as f:
            data = pickle.load(f)
            return data
