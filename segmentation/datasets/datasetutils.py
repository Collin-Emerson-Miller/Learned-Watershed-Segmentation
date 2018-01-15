"""
This utility file is responsible for all dataset utilities
such as downloading and checking.
"""

import urllib2

def download_file(url, filename):
    """
    Downloads file from url and displays progress bar.

    Args:
        url: The URL of the file.
        filename: The name of the file to be saved to disk.

    Returns: None

    """
    u = urllib2.urlopen(url)
    f = open(filename, 'wb')
    meta = u.info()
    file_size = int(meta.getheaders("Content-Length")[0])
    print "Downloading: %s Bytes: %s\n" % (filename, file_size)

    file_size_dl = 0
    block_sz = 8192
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break

        file_size_dl += len(buffer)
        f.write(buffer)
        status = "\r%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
        status = status + chr(8)*(len(status)+1)
        print status,

    f.close()


def get_url_file_size(url):
    """
    Gets the size of a file from a URL in bytes.

    Args:
        url: The URL of the file.

    Returns: The size of the file in bytes.

    """
    u = urllib2.urlopen(url)
    meta = u.info()
    return int(meta.getheaders("Content-Length")[0])