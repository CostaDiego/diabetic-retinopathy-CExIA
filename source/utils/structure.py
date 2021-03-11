from os import path, makedirs

def exists(*args, verbose = False):
    """Check if the list of arguments exists. Works with both files and folders.

    Parameters:
        args: list - List of folder and/or files to be checked

    Returns:
        exists: bool - True if all the files and/or folder exists and False if not.
    """

    for arg in args:
        if  isinstance(arg, list):
            for pth in arg:
                if not path.exists(pth):
                    if verbose:
                        print('{} : Does Not Exists'.format(pth))

                    return False
        else:    
            if not path.exists(arg):
                if verbose:
                    print('{} : Does Not Exists'.format(arg))
                
                return False

    return True

def make_dir(directory:str):
    """
    Check if the directory exists, if don't, create the directory

    Parameters:
        path: str - Path to the folder
    """

    if not path.exists(directory):
        makedirs(directory)