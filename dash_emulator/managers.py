import asyncio
import csv
import os
import pathlib
import random
import signal
import subprocess
import sys
import time
import logging
from typing import Optional, Dict
import tempfile

import aiohttp
import matplotlib.pyplot as plt

from dash_emulator import logger, events, abr, mpd, monitor, config
from dash_emulator.monitor import DownloadProgressMonitor

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class PlayManager(object):
    class State:
        READY = 0
        PLAYING = 1
        STALLING = 2
        STOPPED = 3

    _instance = None

    class DownloadTaskSession(object):
        session_id = 1

        def __init__(self, adaptation_set, url, representation, next_task=None, duration: float = 0, representation_indices=None, segment_index: int = -1):
            self.session_id = PlayManager.DownloadTaskSession.session_id
            PlayManager.DownloadTaskSession.session_id += 1
            self.adaptation_set = adaptation_set  # type: mpd.AdaptationSet
            self.url = url
            self.representation = representation
            self.next_task = next_task
            self.duration = duration
            self.representation_indices = representation_indices
            self.segment_index = segment_index

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = object.__new__(cls, *args, **kwargs)
            cls._instance.inited = False
        return cls._instance

    def __init__(self):
        if not self.inited:
            # Flag shows if the singleton has been initialized
            self.inited = True

            # Properties for configurations
            self.cfg = None  # type: Optional[config.Config]

            # MPD parsed from the target URL
            self.mpd = None  # type: Optional[mpd.MPD]

            # Time
            self.playback_time = 0
            self.playback_start_realtime = 0

            # Asyncio Tasks
            # type: Optional[asyncio.Task]
            self.task_check_buffer_sufficient = None
            # type: Optional[asyncio.Task]
            self.task_update_playback_time = None

            # Playback Control
            self.abr_controller = None  # type: Optional[abr.ABRController]
            self.state = PlayManager.State.READY

            # The representations index of current playback. Key: adaptation set id, value: representation index
            self.representation_indices = dict()  # type: Dict[str, int]

            # Download task sessions
            # type: Dict[str, PlayManager.DownloadTaskSession]
            self.download_task_sessions = dict()

            # Current downloading segment index
            self.segment_index = 0

            # List of stall data
            self.stalls = []  # type: List[Tuple(start_segment_index, end_segment_index stall_duration)]

            # Buffer level data
            self.buffer_level_data = []  # type: List[Tuple(start_segment_index, buffer_level)]

    def switch_state(self, state):
        if state == "READY" or state == PlayManager.State.READY:
            self.state = PlayManager.State.READY
        elif state == "PLAYING" or state == PlayManager.State.PLAYING:
            self.state = PlayManager.State.PLAYING
        elif state == "STALLING" or state == PlayManager.State.STALLING:
            self.state = PlayManager.State.STALLING
        elif state == "STOPPED" or state == PlayManager.State.STOPPED:
            self.state = PlayManager.State.STOPPED
        else:
            log.error("Unknown State: %s" % state)

    @property
    def ready(self):
        return self.state == PlayManager.State.READY

    @property
    def playing(self):
        return self.state == PlayManager.State.PLAYING

    @property
    def stalling(self):
        return self.state == PlayManager.State.STALLING

    @property
    def stopped(self):
        return self.state == PlayManager.State.STOPPED

    @property
    def buffer_level(self):
        return (monitor.BufferMonitor().buffer - self.playback_time) * 1000

    async def check_buffer_sufficient(self):
        while True:
            if self.buffer_level > 0:
                self.buffer_level_data.append((self.segment_index, self.buffer_level / 1000))
                log.info("Buffer level sufficient: %.1f seconds" %
                         (self.buffer_level / 1000))
                await asyncio.sleep(min(1, self.buffer_level / 1000))
            else:
                break
        if self.mpd.mediaPresentationDuration <= self.playback_time:
            await events.EventBridge().trigger(events.Events.End)
        else:
            await events.EventBridge().trigger(events.Events.Stall)

    async def update_current_time(self):
        while True:
            await asyncio.sleep(self.cfg.update_interval)
            self.playback_time += self.cfg.update_interval

    def clear_download_task_sessions(self):
        self.download_task_sessions.clear()

    def redo_session_low_quality(self, adaptation_set: mpd.AdaptationSet):
        """
        Re-request same segment (for one adaptation set) at lowest quality.
        """

        self.representation_indices[adaptation_set.id] = 0  # lowest quality
        representation = adaptation_set.representations[0]
        segment_number = self.segment_index + representation.startNumber
        segment_download_session = PlayManager.DownloadTaskSession(adaptation_set,
                                                                   representation.urls[segment_number],
                                                                   representation,
                                                                   None,
                                                                   representation.durations[segment_number],
                                                                   representation_indices=self.representation_indices,
                                                                   segment_index=self.segment_index)
        if not representation.is_inited:
            init_download_session = PlayManager.DownloadTaskSession(adaptation_set,
                                                                    representation.initialization_url,
                                                                    representation,
                                                                    segment_download_session,
                                                                    0,
                                                                    representation_indices=self.representation_indices,
                                                                    segment_index=self.segment_index)
            representation.is_inited = True
            self.download_task_sessions[adaptation_set.id] = init_download_session
        else:
            self.download_task_sessions[adaptation_set.id] = segment_download_session

    def create_session_urls(self):
        """
        Create download_task_sessions for current segment_index
        :return:
        """
        for adaptation_set in self.mpd.adaptationSets.values():
            representation_index = self.representation_indices[adaptation_set.id]
            representation = adaptation_set.representations[representation_index]
            segment_number = self.segment_index + representation.startNumber
            segment_download_session = PlayManager.DownloadTaskSession(adaptation_set,
                                                                       representation.urls[segment_number],
                                                                       representation,
                                                                       None,
                                                                       representation.durations[segment_number],
                                                                       self.representation_indices,
                                                                       segment_index=self.segment_index)

            if not representation.is_inited:
                init_download_session = PlayManager.DownloadTaskSession(adaptation_set,
                                                                        representation.initialization_url,
                                                                        representation,
                                                                        segment_download_session,
                                                                        0,
                                                                        self.representation_indices,
                                                                        segment_index=self.segment_index)
                representation.is_inited = True
                self.download_task_sessions[adaptation_set.id] = init_download_session
            else:
                self.download_task_sessions[adaptation_set.id] = segment_download_session
        

    def init(self, cfg, mpd: mpd.MPD):
        self.cfg = cfg  # type: config.Config
        self.mpd = mpd

        for adaptation_set in mpd.adaptationSets.values():
            # Init the representation of each adaptation set to -1
            self.representation_indices[adaptation_set.id] = -1

        self.abr_controller = abr.ABRController()

        # Play immediately
        async def can_play():
            log.info("The player is ready to play")
            await events.EventBridge().trigger(events.Events.Play)

        events.EventBridge().add_listener(events.Events.CanPlay, can_play)

        async def play():
            log.info("Video playback started")
            self.playback_start_realtime = time.time()

            try:
                self.task_update_playback_time = asyncio.create_task(
                    self.update_current_time())
            except AttributeError:
                loop = asyncio.get_event_loop()
                self.task_update_playback_time = loop.create_task(
                    self.update_current_time())

            try:
                self.task_check_buffer_sufficient = asyncio.create_task(
                    self.check_buffer_sufficient())
            except AttributeError:
                loop = asyncio.get_event_loop()
                self.task_check_buffer_sufficient = loop.create_task(
                    self.check_buffer_sufficient())
            self.switch_state("PLAYING")

        events.EventBridge().add_listener(events.Events.Play, play)

        async def stall():
            if self.task_update_playback_time is not None:
                self.task_update_playback_time.cancel()
            if self.task_check_buffer_sufficient is not None:
                self.task_check_buffer_sufficient.cancel()

            log.debug("Stall happened")
            self.switch_state("STALLING")
            before_stall = time.time()
            before_stall_segment_index = self.segment_index
            while True:
                await asyncio.sleep(self.cfg.update_interval)
                if monitor.BufferMonitor().buffer - self.playback_time > self.mpd.minBufferTime:
                    break
            
            stall_duration = time.time() - before_stall
            self.stalls.append( (before_stall_segment_index, self.segment_index, stall_duration) )
            log.debug("Stall ends, duration: %.3f" % stall_duration)

            await events.EventBridge().trigger(events.Events.Play)

        events.EventBridge().add_listener(events.Events.Stall, stall)

        async def download_repeat_start(session: PlayManager.DownloadTaskSession, *args, **kwargs):
            adaptation_set = session.adaptation_set
            # start downloading a single adaptation set
            log.debug(
                f"Repeat download for adaptation_set {adaptation_set.id}")

            # setting session @ lowest quality
            self.redo_session_low_quality(adaptation_set)
            await events.EventBridge().trigger(events.Events.DownloadStart, session=session)
        events.EventBridge().add_listener(
            events.Events.RedoTileAtLowest, download_repeat_start)

        async def download_start():
            # Start downloading all adaptation sets
            self.representation_indices = self.abr_controller.calculate_next_segment(monitor.SpeedMonitor().get_speed(),
                                                                                     self.segment_index,
                                                                                     self.representation_indices,
                                                                                     self.mpd.adaptationSets)
            self.create_session_urls()

            for adaptation_set_id, session in self.download_task_sessions.items():
                await events.EventBridge().trigger(events.Events.DownloadStart, session=session)

        events.EventBridge().add_listener(events.Events.MPDParseComplete, download_start)

        async def download_next(session: PlayManager.DownloadTaskSession, *args, **kwargs):
            if session.next_task is None:
                del self.download_task_sessions[session.adaptation_set.id]
            else:
                self.download_task_sessions[session.adaptation_set.id] = session.next_task
                await events.EventBridge().trigger(events.Events.DownloadStart, session=session.next_task)
                return
            if len(self.download_task_sessions) == 0:
                await events.EventBridge().trigger(events.Events.SegmentDownloadComplete,
                                                   segment_index=self.segment_index, duration=session.duration)
                
                # Record avg bandwidth for previous segment download
                avg_bandwidth = monitor.SpeedMonitor().get_speed() # Total bandwidth used to download all tiles in bps
                log.debug(f"Adding to avg bandwidth {avg_bandwidth} at [{self.segment_index}]")
                DownloadManager().download_record[self.segment_index] = [avg_bandwidth, DownloadManager().current_representations]
                DownloadManager().current_representations = []  # Reset current representations for next round

                self.segment_index += 1
                self.representation_indices = self.abr_controller.calculate_next_segment(
                    monitor.SpeedMonitor().get_speed(),
                    self.segment_index,
                    self.representation_indices,
                    self.mpd.adaptationSets)

                try:
                    self.create_session_urls()
                except IndexError as e:
                    await events.EventBridge().trigger(events.Events.DownloadEnd)
                else:
                    for adaptation_set_id, session in self.download_task_sessions.items():
                        await events.EventBridge().trigger(events.Events.DownloadStart, session=session)

        async def check_canplay(*args, **kwargs):
            if self.ready and self.buffer_level > self.mpd.minBufferTime:
                await events.EventBridge().trigger(events.Events.CanPlay)

        events.EventBridge().add_listener(events.Events.DownloadComplete, download_next)
        events.EventBridge().add_listener(events.Events.BufferUpdated, check_canplay)

        async def ctrl_c_handler():
            print("Fast-forward to the end")
            # Change current time to 0.5 seconds before the end
            self.playback_time = monitor.BufferMonitor().buffer - 0.5
            asyncio.get_event_loop().remove_signal_handler(signal.SIGINT)

        async def download_end():
            await monitor.SpeedMonitor().stop()
            await asyncio.sleep(0.5)

            loop = asyncio.get_event_loop()
            if not sys.platform.startswith('win32'):
                if sys.version_info.minor < 7:
                    loop.add_signal_handler(
                        signal.SIGINT, lambda: asyncio.ensure_future(ctrl_c_handler()))
                else:
                    loop.add_signal_handler(
                        signal.SIGINT, lambda: asyncio.create_task(ctrl_c_handler()))
                print("You can press Ctrl-C to fastforward to the end of playback")

        events.EventBridge().add_listener(events.Events.DownloadEnd, download_end)

        async def buffer_updated(buffer, *args, **kwargs):
            log.info("Current Buffer Level: %.3f" % self.buffer_level)

        async def plot():
            output_folder = os.path.join(self.cfg.args['output'], "figures") if self.cfg.args['output'] else os.path.join(os.curdir, "figures")
            output_folder = pathlib.Path(output_folder).absolute()
            output_folder.mkdir(parents=True, exist_ok=True)

            num_representatinons = len(DownloadManager().current_representations)
            for i in range(0, num_representatinons):
                # Durations of segments
                durations = DownloadManager().current_representations[i][1].durations
                start_num = DownloadManager().current_representations[i][1].startNumber
                fig = plt.figure()
                plt.plot([i for i in range(start_num, len(durations))], durations[start_num:])
                plt.xlabel("Segments")
                plt.ylabel("Durations (sec)")
                plt.title("Durations of each segment")
                fig.savefig(os.path.join(output_folder, f"segment-durations{i}.pdf"))
                plt.close()

                # Download bandwidth of each segment
                fig = plt.figure()
                inds = [i for i in sorted(DownloadManager()._bandwidth_segmentwise.keys())]
                bws = [DownloadManager()._bandwidth_segmentwise[i] for i in inds]
                plt.plot(inds, bws)
                plt.xlabel("Segments")
                plt.ylabel("Bandwidth (bps)")
                plt.title("Bandwidth of downloading each segment")
                fig.savefig(os.path.join(output_folder, "segment-download-bandwidth.pdf"))
                plt.close()

        async def exit_program():
            print("Prepare to exit the program")
            events.EventBridge().over = True

        async def validate_output_path() -> None:
            """
            This function is used to validate the output path
            It will create the folder if it doesn't exist
            It will prompt a message to ask for deleting everything in the folder
            """
            output_path = self.cfg.args['output']
            path = pathlib.Path(output_path)
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
            files = [i for i in path.glob("*")]
            delete_choice = self.cfg.args['y']
            if len(files) > 0:
                delete_choice = delete_choice or (
                    input("Do you want to delete files in the output folder? (y/N)") == 'y')
            if delete_choice:
                # shutil.rmtree(path.absolute())
                subprocess.call(['rm', '-rf', str(path) + "/*"])
                path.mkdir(parents=True, exist_ok=True)

        async def output() -> None:
            """
            Generate output reports and videos
            """
            await validate_output_path()
            output_path = self.cfg.args['output']

            # Reports
            dr = DownloadManager().download_record
            with open(os.path.join(output_path, 'results.csv'), 'w') as f:
                writer_stalls = csv.DictWriter(f, dialect='excel', fieldnames=["start_segment_index", "end_segment_index", "duration"])
                writer_stalls.writeheader()

                for stall_tuple in self.stalls:
                    record_stalls = {
                        "start_segment_index" : stall_tuple[0], 
                        "end_segment_index" : stall_tuple[1],
                        "duration" : stall_tuple[2]
                    }
                    writer_stalls.writerow(record_stalls)
                writer_stalls.writerow({})


                writer_buffer_level = csv.DictWriter(f, dialect='excel', fieldnames=["segment_index", "buffer_level_seconds"])
                writer_buffer_level.writeheader()

                for buffer_tuple in self.buffer_level_data:
                    record_buffer = {
                        "segment_index" : buffer_tuple[0],
                        "buffer_level_seconds" : buffer_tuple[1]
                    }
                    writer_buffer_level.writerow(record_buffer)
                writer_buffer_level.writerow({})


                writer = csv.DictWriter(f, dialect='excel', fieldnames=["segment_index", "total_avg_bandwidth", "avg_bandwidth_per_tile", 
                                        "ABR_avg_bandwidth_per_tile", "filename", "id", "bitrate",
                                        "width", "height", "mime", "codec", "bandwidth_difference"])
                writer.writeheader()
                for ind in dr.keys():
                    bws = dr[ind][0]
                    representation_tuples = dr[ind][1]

                    avg_bandwidth_per_tile = bws / len(self.mpd.adaptationSets)
                    ABR_avg_bandwidth_per_tile = avg_bandwidth_per_tile * self.cfg.bandwidth_fraction
                    record1 = {
                        "segment_index" : ind,
                        "total_avg_bandwidth" : bws,
                        "avg_bandwidth_per_tile" : avg_bandwidth_per_tile,
                        "ABR_avg_bandwidth_per_tile": ABR_avg_bandwidth_per_tile
                    }
                    writer.writerow(record1)
                    bws_sum = 0
                    bws_avg_counter = 0
                    for r_tuple in representation_tuples:
                        tile_name = r_tuple[0]
                        representation = r_tuple[1]
                        bws_sum += representation.bandwidth
                        bws_avg_counter += 1
                        record2 = {
                            "filename": tile_name,
                            "id": representation.id,
                            "bitrate": representation.bandwidth,
                            "width": representation.width,
                            "height": representation.height,
                            "mime": representation.mime,
                            "codec": representation.codec
                        }
                        writer.writerow(record2)

                    theoretical_avg_bandwidth = bws_sum / bws_avg_counter   # Per tile
                    bandwidth_difference = ABR_avg_bandwidth_per_tile - theoretical_avg_bandwidth
                    record_avg = {
                        "bitrate" : theoretical_avg_bandwidth,
                        "bandwidth_difference" : bandwidth_difference
                    }
                    writer.writerow(record_avg)

                # Merge segments into a complete one
                # tmp_output_dir_name = '-playback-merge-%05d' % random.randint(1, 99999)
                # with tempfile.TemporaryDirectory(suffix=tmp_output_dir_name) as tmp_output_path:
                #     # tmp_output_path = "temp"
                #     # os.mkdir(tmp_output_path)
                #     log.info(f'Creating temporary directory: {tmp_output_path}')
                #     segment_list_file_name = 'segment-list.txt'
                #     segment_list_file_path = os.path.join(tmp_output_path, segment_list_file_name)
                #     log.info(f'Creating temporary file: {segment_list_file_path}')
                #     target_output_path = "%s/%s" % (output_path, 'playback.mp4')

                #     for (segment_name, representation), ind, bw in zip(DownloadManager().download_record, seg_inds, bws):
                #         with open('%s/%s' % (output_path, 'merge-segment-%d.mp4' % ind), 'wb') as f:
                #             subprocess.call(['cat', f"{output_path}{os.path.sep}{representation.init_filename}",
                #                                 f"{output_path}{os.path.sep}{segment_name}"], stdout=f)
                #         tmp_segment_path = f'{tmp_output_path}{os.path.sep}segment-{ind}.mp4'
                #         subprocess.call(
                #             ['ffmpeg', '-i', f"{output_path}{os.path.sep}merge-segment-{ind}.mp4", '-vcodec', 'libx264',
                #             '-vf', 'scale=1920:1080', tmp_segment_path], stdout=subprocess.DEVNULL,
                #             stderr=subprocess.DEVNULL)
                #         with open(segment_list_file_path, 'a') as f:
                #             subprocess.call(['echo', f'file {tmp_segment_path}'], stdout=f)
                #             f.flush()

            # Merge segments into a complete one
            # print(target_output_path)
            # subprocess.call(
            #     ['ffmpeg', '-f', 'concat', '-safe', '0', '-i',
            #         segment_list_file_path, '-c', 'copy', target_output_path],
            #     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if self.cfg.args['output'] is not None:
            events.EventBridge().add_listener(events.Events.End, output)
        if self.cfg.args['plot']:
            events.EventBridge().add_listener(events.Events.End, plot)
        events.EventBridge().add_listener(events.Events.End, exit_program)


class DownloadManager(object):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = object.__new__(cls, *args, **kwargs)
            cls._instance.inited = False
        return cls._instance

    def __init__(self):
        if not self.inited:
            self.inited = True
            self.cfg = None  # type: Optional[config.Config]

            self.tile_download_times = {} # type: Dict[str, List[float]]
            self.tile_percentages_received = {} # type: Dict[str, List[float]]
            self.segment_buffer_levels = [] # type: List[float]

            # from main because there was an error on line 398
            # This property records all downloaded segments and its corresponding representation
            # Each tuple in the list represents a segment
            # Each tuple contains 2 elements: filename of the segment, corresponding representation
            #TODO: change this to a 2D list of tples: List[segment_index, bandwidth, List[Tuple[str, mpd.Representation]]]
                                        # Might as well merge this and _bandwidth_segmentwise? Or, move _bandwidth_segmentwise back into the PlayManager
            self.download_record = {}  # type: {"segment_index" : List["avg_bandwidth", List[Tuple[str, mpd.Representation]]}
            self.current_representations = []  # type: List[MPD.Representation]

    async def record_buffer_level(self, *args, **kwargs):
        print('recording buffer level')
        buffer_level = monitor.BufferMonitor().buffer
        self.segment_buffer_levels.append(buffer_level)
    
    async def dump_results(self, *args, **kwargs):
        """
        Dumps all playback statistics to the console. Called when playback is over.
        """
        print('dumping results')
        csvLogger = logger.getLogger("csv")
        if (self.cfg.args['output']):
            consoleHandler = logging.StreamHandler(stream=open(os.path.join(self.cfg.args['output'], "dump_results.csv"), "w+", encoding="utf-8"))
            consoleHandler.setLevel(level=logging.INFO)
            formatter = logger.CsvFormatter()
            consoleHandler.setFormatter(formatter)
            csvLogger.addHandler(consoleHandler)
            csvLogger.propagate = False

        csvLogger.info('\n\n**Results**\n\n')

        # logging 'percentage downloaded' statistics for all tile-segments
        csvLogger.info('Percentage of bytes received for each tile:\n')
        segment_index = 0
        num_segments = len(self.segment_buffer_levels)
        csvLogger.debug(f'Number of segments \t {num_segments}')
        num_tiles = len(self.tile_download_times)
        tile_percent_sums = num_tiles * [0.0]
        csvLogger.info(f'segment_index')
        for i in range(num_tiles):
            csvLogger.info(f'Tile #{i}\t')
        csvLogger.info(' \t Overall\n')
        
        while segment_index < num_segments:
            csvLogger.info(f'{segment_index} \t \t ')
            sum_of_percents = 0
            for i in range(num_tiles):
                percentage = self.tile_percentages_received[str(i)][segment_index]
                tile_percent_sums[i] += percentage
                sum_of_percents += percentage
                csvLogger.info(f'{percentage},\t')
            # write the average
            average = float(sum_of_percents / num_tiles)
            csvLogger.info(f' \t \t {average}\n')
            segment_index += 1
        # finishing with overall averages
        csvLogger.info('Average\t')
        cum_sum = 0
        for i in range(num_tiles):
            average = float(tile_percent_sums[i] / num_segments)
            cum_sum += average
            csvLogger.info(f'{average},\t')
        overall_average = float(cum_sum / num_tiles)
        csvLogger.info(f'\t\t {overall_average}\n')

        csvLogger.info("\nDownload record:")
        dr = DownloadManager().download_record
        for ind in dr.keys():
           csvLogger.info(f"Segment Index: {ind} \t Avg bandwidth: {dr[ind][0]} \n")
           for r_tuple in dr[ind][1]:
               name = r_tuple[0]
               r = r_tuple[1]
               csvLogger.info(f"Representation: \t name: {name} \t id: {r.id} \t url: {r.baseurl} \t bandwidth: {r.bandwidth}")

    async def download(self, url, download_progress_monitor) -> None:
        """
        Download the file of url and save it if `output` shows in the args
        """
        download_progress_monitor.start_time = time.time()

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                output = self.cfg.args['output']
                if output is not None:
                    f = open(output + os.path.sep + url.split('/')[-1], 'wb')
                download_progress_monitor.segment_length = int(resp.headers["Content-Length"])
                while True:
                    chunk = await resp.content.read(self.cfg.chunk_size)
                    if not chunk:
                        if output is not None:
                            f.close()
                        break
                    monitor.SpeedMonitor().downloaded += (len(chunk) * 8)
                    download_progress_monitor.downloaded += len(chunk)
                    if output is not None:
                        f.write(chunk)
                # once here, the download has finished; save data to be logged post-experiment
                if download_progress_monitor.session.duration == 0:
                    # true for init stream files, which we don't care about
                    return
                
                adaptation_set_id = download_progress_monitor.session.adaptation_set.id
                end_time = time.time()
                time_to_receive_tile = end_time - download_progress_monitor.start_time
                percent_received = float(download_progress_monitor.downloaded / download_progress_monitor.segment_length)

                # record 'time to download (s)' and 'bytes received (%)'
                self.tile_download_times[adaptation_set_id].append(time_to_receive_tile)
                self.tile_percentages_received[adaptation_set_id].append(percent_received)

    def init(self, cfg: config.Config, num_tiles: int = 0) -> None:
        """
        Init the download manager, including add callbacks to events
        """
        self.cfg = cfg
        # prep data structures for use later
        for i in range(num_tiles):
            self.tile_download_times[str(i)] = []
            self.tile_percentages_received[str(i)] = [] 


        async def start_download(session: PlayManager.DownloadTaskSession):
            # if self.segment_index >= self.segment_num:
            #     await events.EventBridge().trigger(events.Events.DownloadEnd)
            #     return

            # If the init file hasn't been downloaded for this representation, download that first

            # if not self.representation.is_inited:
            #     url = self.representation.initialization
            #
            #     try:
            #         task = asyncio.create_task(self.download(url))
            #     except AttributeError:
            #         loop = asyncio.get_event_loop()
            #         task = loop.create_task(self.download(url))
            #
            #     await task
            #     self.representation.is_inited = True
            #     self.representation.init_filename = url.split('/')[-1]
            #     await events.EventBridge().trigger(events.Events.InitializationDownloadComplete)
            #     log.info("Download initialization for representation %s" % self.representation.id)

            url = session.url

            assert url is not None

            download_progress_monitor = DownloadProgressMonitor(
                self.cfg, session)

            # Download the segment
            try:
                task = asyncio.create_task(
                    self.download(url, download_progress_monitor))
            except AttributeError:
                loop = asyncio.get_event_loop()
                task = loop.create_task(self.download(
                    url, download_progress_monitor))

            download_progress_monitor.task = task

            try:
                monitor_task = asyncio.create_task(
                    download_progress_monitor.start())  # type: asyncio.Task
            except AttributeError:
                loop = asyncio.get_event_loop()
                monitor_task = loop.create_task(
                    download_progress_monitor.start())


            self.current_representations.append( (session.url.split('/')[-1], session.representation) )
            
            try:
                await task
            except asyncio.CancelledError:
                print("Partial Received")
            monitor_task.cancel()
            await events.EventBridge().trigger(events.Events.DownloadComplete, session=session)

        events.EventBridge().add_listener(events.Events.DownloadStart, start_download)
        events.EventBridge().add_listener(events.Events.SegmentDownloadComplete, self.record_buffer_level)
        events.EventBridge().add_listener(events.Events.End, self.dump_results)
