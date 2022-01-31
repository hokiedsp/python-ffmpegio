from .. import configure, threading, utils, ffmpegprocess

__all__ = ["AviMediaReader"]


class AviMediaReader:
    """Read video frames

    :param *urls: URLs of the media files to read.
    :type *urls: tuple(str)
    :param streams: list of file + stream specifiers or filtergraph label to output, alias of `map` option,
                    defaults to None, which outputs at most one video and one audio, selected by FFmpeg
    :type streams: seq(str), optional
    :param progress: progress callback function, defaults to None
    :type progress: callable object, optional
    :param show_log: True to show FFmpeg log messages on the console,
                     defaults to None (no show/capture)
                     Ignored if stream format must be retrieved automatically.
    :type show_log: bool, optional
    :param use_ya8: True if piped video streams uses `ya8` pix_fmt instead of `gray16le`, default to None
    :type use_ya8: bool, optional
    :param \\**options: FFmpeg options, append '_in[input_url_id]' for input option names for specific
                        input url or '_in' to be applied to all inputs. The url-specific option gets the
                        preference (see :doc:`options` for custom options)
    :type \\**options: dict, optional

    :return: frame rate and video frame data (dims: time x rows x cols x pix_comps)
    :rtype: (`fractions.Fraction`, `numpy.ndarray`)

    Note: Only pass in multiple urls to implement complex filtergraph. It's significantly faster to run
          `ffmpegio.video.read()` for each url.


    Unlike :py:mod:`video` and :py:mod:`image`, video pixel formats are not autodetected. If output
    'pix_fmt' option is not explicitly set, 'rgb24' is used.

    For audio streams, if 'sample_fmt' output option is not specified, 's16le'.


    streams = ['0:v:0','1:a:3'] # pick 1st file's 1st video stream and 2nd file's 4th audio stream

    """

    readable = True
    writable = False
    multi_read = True
    multi_write = False

    def __init__(
        self,
        *urls,
        ref_stream=None,
        blocksize=None,
        progress=None,
        show_log=None,
        queuesize=0,
        **options
    ):

        self.ref_stream = ref_stream
        #:str: specifier of reference output stream for iterator
        self.blocksize = blocksize or 0
        #:int: if >0 number of samples of reference stream to include in each read; <=0 one chunk per read

        ninputs = len(urls)
        if not ninputs:
            raise ValueError("At least one URL must be given.")

        # separate the options
        spec_inopts = utils.pop_extra_options_multi(options, r"_in(\d+)$")
        inopts = utils.pop_extra_options(options, "_in")

        # create a new FFmpeg dict
        args = configure.empty()
        configure.add_url(args, "output", "-", options)  # add piped output
        for i, url in enumerate(urls):  # add inputs
            # check url (must be url and not fileobj)
            configure.check_url(url, nodata=True, nofileobj=True)
            configure.add_url(args, "input", url, {*inopts, *spec_inopts.get(i, {})})

        # configure output options
        use_ya8 = configure.finalize_media_read_opts(args)

        self._reader = threading.AviReaderThread(use_ya8, queuesize)

        # create logger without assigning the source stream
        self._logger = threading.LoggerThread(None, show_log)

        # start FFmpeg
        self._proc = ffmpegprocess.Popen(
            args,
            progress=progress,
            capture_log=True,
            close_stdin=True,
            close_stdout=False,
            close_stderr=False,
        )

        # start the reader thrad
        self._reader.start(self._proc.stdout)

        # set the log source and start the logger
        self._logger.stderr = self._proc.stderr
        self._logger.start()

    def types(self):
        """:dict(str:str): media type associated with the streams (key)"""
        self._reader.wait()
        ts = {"v": "video", "a": "audio"}
        return {v["spec"]: ts[v["type"]] for v in self._reader.streams.values()}

    def rates(self):
        """:dict(str:int|Fraction): sample or frame rates associated with the streams (key)"""
        self._reader.wait()
        rates = self._reader.rates
        return {v["spec"]: rates[k] for k, v in self._reader.streams.items()}

    def dtypes(self):
        """:dict(str:str): numpy dtypes associated with the streams (key)"""
        self._reader.wait()
        return {v["spec"]: v["dtype"] for v in self._reader.streams.values()}

    def shapes(self):
        """:dict(str:tuple(int)): base array shape associated with the streams (key)"""
        self._reader.wait()
        return {v["spec"]: v["shape"] for v in self._reader.streams.values()}

    def get_stream_info(self, spec):
        id = self._reader.find_id(spec)
        return self._reader.streams[id]

    def close(self):
        """Flush and close this stream. This method has no effect if the stream is already
            closed. Once the stream is closed, any read operation on the stream will raise
            a ValueError.

        As a convenience, it is allowed to call this method more than once; only the first call,
        however, will have an effect.

        """
        self._proc.stdout.close()
        self._proc.stderr.close()
        try:
            self._proc.terminate()
        except:
            pass
        self._reader.join()
        self._logger.join()

    @property
    def closed(self):
        """:bool: True if the FFmpeg has been terminated."""
        return self._proc.poll() is not None

    @property
    def lasterror(self):
        """:FFmpegError: TODO Last error FFmpeg posted"""
        if self._proc.poll():
            return self._logger.Exception()
        else:
            return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return (
                self._reader.read(self.blocksize, self.ref_stream)
                if self.blocksize > 0
                else self._reader.readchunk()
            )
        except:
            raise StopIteration

    def readlog(self, n=None):
        if n is not None:
            self._logger.index(n)
        with self._logger._newline_mutex:
            return "\n".join(self._logger.logs or self._logger.logs[:n])

    def readnext(self, timeout=None):
        return self._reader.readchunk(timeout)

    def read(self, n=-1, ref_stream=None, timeout=None):
        """Read and return numpy.ndarray with up to n frames/samples. If
        the argument is omitted, None, or negative, data is read and
        returned until EOF is reached. An empty bytes object is returned
        if the stream is already at EOF.

        If the argument is positive, and the underlying raw stream is not
        interactive, multiple raw reads may be issued to satisfy the byte
        count (unless EOF is reached first). But for interactive raw streams,
        at most one raw read will be issued, and a short result does not
        imply that EOF is imminent.

        A BlockingIOError is raised if the underlying raw stream is in non
        blocking-mode, and has no data available at the moment."""

        return self._reader.read(n, ref_stream, timeout)

    def readall(self, timeout=None):
        return self._reader.readall(timeout)
