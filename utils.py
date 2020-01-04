import copy
import logging
import colorama
from math import floor

def colourize_string(string, colour):
    return '{color_begin}{string}{color_end}'.format(
        string=string,
        color_begin=colour,
        color_end=colorama.Style.RESET_ALL)

def initialize_logger_format(logger):
    """
    Initialize the specified logger with a coloured format.
    """

    # specify colors for different logging levels
    LOG_COLORS = {
        logging.FATAL: colorama.Fore.LIGHTRED_EX,
        logging.ERROR: colorama.Fore.RED,
        logging.WARNING: colorama.Fore.YELLOW,
        logging.DEBUG: colorama.Fore.LIGHTWHITE_EX
    }

    LOG_LEVEL_FORMATS = {
        logging.INFO: '%(message)s'
    }

    class CustomFormatter(logging.Formatter):
        def format(self, record, *args, **kwargs):
            # if the corresponding logger has children, they may receive modified
            # record, so we want to keep it intact
            new_record = copy.copy(record)
            if new_record.levelno in LOG_COLORS:
                # we want levelname to be in different color, so let's modify it
                new_record.levelname = "{color_begin}{level}{color_end}".format(
                    level=new_record.levelname,
                    color_begin=LOG_COLORS[new_record.levelno],
                    color_end=colorama.Style.RESET_ALL,
                )

            original_format = self._style._fmt
            self._style._fmt = LOG_LEVEL_FORMATS.get(record.levelno, original_format)

            # now we can let standart formatting take care of the rest
            result = super(CustomFormatter, self).format(new_record, *args, **kwargs)

            self._style._fmt = original_format
            return result

    handler = logging.StreamHandler()
    handler.setFormatter(CustomFormatter('%(levelname)s: %(message)s'))
    logger.addHandler(handler)

def init_logger():
    logger = logging.getLogger(__name__)
    initialize_logger_format(logger)

    return logger

def get_geo_stepsizes(points, chunk_width, chunk_height):
    '''
    Get longitude and latitude stepsizes.

    :returns:
        The longitude and latitude stepsize respectively.
    '''

    min_longitude, max_longitude = min(points, key=lambda i: i.x).x, max(points, key=lambda i: i.x).x
    longitude_stepsize = (max_longitude - min_longitude) / chunk_width

    min_latitude, max_latitude = min(points, key=lambda i: i.y).y, max(points, key=lambda i: i.y).y
    latitude_stepsize = (max_latitude - min_latitude) / chunk_height

    return longitude_stepsize, latitude_stepsize

def partition(points, chunk_width, chunk_height):
    '''
    Partition a list of points into chunks.
    '''

    min_longitude, max_longitude = min(points, key=lambda i: i.x).x, max(points, key=lambda i: i.x).x
    min_latitude, max_latitude = min(points, key=lambda i: i.y).y, max(points, key=lambda i: i.y).y
    longitude_stepsize, latitude_stepsize = get_geo_stepsizes(points, chunk_width, chunk_height)
    
    chunks = [[[] for _ in range(chunk_height)] for _ in range(chunk_width)]
    for point in points:
        # Handle edge case for when a point lies on the
        # maximum longitudinal boundary or the minimum 
        # latitudinal bondary
        if point.x == max_longitude:
            chunk_x = chunk_width - 1
        else:
            chunk_x = floor((point.x - min_longitude) / longitude_stepsize)

        if point.y == min_latitude:
            chunk_y = chunk_height - 1
        else:
            chunk_y = floor((max_latitude - point.y) / latitude_stepsize)

        chunks[chunk_x][chunk_y].append(point)

    return chunks