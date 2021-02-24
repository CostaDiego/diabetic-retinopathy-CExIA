from os import path, register_at_fork

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