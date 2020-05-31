def get_pr(file_name):
    """
    Imports the pr value from the given file
    :param file_name: file which contains the pr value
    :return: evidence value as an array
    """
    try:
        with open(file_name) as FileObj:
            for lines in FileObj.readlines():
                if lines != '\n':
                    values = lines.split()
        return float(values[0])
    except:
        print("The directory for the evidence file is not correct, please check.")
        exit()
