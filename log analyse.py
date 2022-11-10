from dataclasses import dataclass
from datetime import datetime, date, time, timedelta, tzinfo, timezone
from dateutil.relativedelta import relativedelta
from dateutil import tz
from tzinfo_examples import HOUR, Pacific
from typing import List, Tuple, Type
from typing_extensions import Self
from pathlib import Path, PurePath
from io import SEEK_END
from re import search
from itertools import islice
from json import load as jsonload
from pipe import select, where
from bs4 import BeautifulSoup, SoupStrainer
from requests import get
from requests.exceptions import MissingSchema
from collections import namedtuple
from math import ceil
import logging
import logging.config
import os
import yaml
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_LEVEL = logging.INFO
def log_setup(log_cfg_path='log cfg.yaml'):
    if os.path.exists(log_cfg_path):
        with open(log_cfg_path, 'rt') as cfg_file:
            try:
                config = yaml.safe_load(cfg_file.read())
                logging.config.dictConfig(config)
            except Exception as e:
                print('Error with file, using Default logging')
                logging.basicConfig(level=DEFAULT_LEVEL)
    else:
        logging.basicConfig(level=DEFAULT_LEVEL)
        print('Config file not found, using Default logging')

log_setup()
logger = logging.getLogger(__name__)

def url_validate(url: str) -> get:
    try:
        _img = get(url)
    except MissingSchema:
        return None
    except Exception:
        return None
    else:
        return _img


@dataclass
class Skill:
    name:str
    parent:Type[Self]
    Characteristics:list[Type[Self]]

skills = []
with open("skills.json", "r") as file:      
    data = jsonload(file)
    for entry in islice(data["skills"], 0, None):
        skills.append(Skill(**entry))


@dataclass
class Action:
    name:str
    skill:Type[Skill]

actions = []
with open("actions.json", "r") as file:
    data = jsonload(file)
    for entry in islice(data["actions"], 0, None):
        actions.append(Action(**entry))


@dataclass
class StartMessage:
    line:str
    action:Type[Action]
start_messages = []


@dataclass
class EndMessage:
    line:str
    action:Type[Action]
end_messages = []


with open("event messages.json", 'r') as file:      
    data = jsonload(file)
    for entry in islice(data["start_messages"], 0, None):
        start_messages.append(StartMessage(**entry))
    for entry in islice(data["end_messages"], 0, None):
        end_messages.append(EndMessage(**entry))


action_attempts = []

@dataclass
class ActionAttempt():
    """A single attempted action"""
    start_time:datetime
    end_time:datetime
    action:Type[Action]
    isRarity:bool=False


EventStartMsg = namedtuple("EventStartMsg", "datetime action")

class EventLog():

    def __init__(self, start_datetime: datetime, end_datetime: datetime, html_path: str=None, file_path: Path=None):
        self.file_path = file_path
        self.tell_pos = None
        self.last_modified = None
        self.web_file = None
        self.start_datetime = start_datetime

        self.tz_org = tz.gettz()

        self.my_std = None
        self.my_dst = None
        for v in self.tz_org._tznames:
            if "Standard" in v:
                self.my_std = timezone(self.tz_org._std_offset, name='std')
            elif "Daylight" in v:
                self.my_dst = timezone(self.tz_org._dst_offset, name='dst') 

        self.start_datetime = start_datetime.astimezone(self.tz_org).astimezone(tz.UTC)
        self.end_datetime = end_datetime.astimezone(self.tz_org).astimezone(tz.UTC)
        if html_path != None:
            self.web_file = url_validate(html_path)
        elif file_path != None:
            if not isinstance(self.file_path, PurePath):
                logger.critical('combat log file path is not a Path object')
                return
            if not self.file_path.exists() or not self.file_path.is_file():
                logger.critical(
                    'file exists..{},  file is a file..{}, {}'.format(self.file_path.exists(), 
                    self.file_path.is_file(), str(self.file_path)))
                return
        else:
            logger.error('no valid log data source provided.')
            return
        self._get_start_index()
        
    def _get_start_index(self):
        # Find line index that has a time stamp which is at most equal to the start date.
        self.start_line_index = 0
        _time = _date = None
        self.last_datetime = None
        self._reset_line_gen()
        for i, line in enumerate(self.line_gen):
            if line == "\n":
                continue
            _date, _time = self._time_stamp_import(line, _date, _time)
            if _time is None:
                continue
            if _time is not None and _date is None:
                logger.critical(f'No date found. line: {line}. index: {self.start_line_index}')
                return
            _datetime = datetime.combine(_date, _time).astimezone(self.tz_org)
            if i < 4 and self.last_datetime == None:
                if tz.datetime_ambiguous(_datetime, self.tz_org):
                    logger.critical('potential ambiguous time because of DST')
                    return
                self.last_datetime = _datetime.astimezone(tz.UTC)
            _search = search(r"Logging started ", line)
            if _search is not None:
                continue
            if tz.datetime_ambiguous(datetime.combine(_date, _time), self.tz_org) \
                and _datetime.astimezone(tz.UTC) < self.last_datetime:
                    # Fall back,  standard time.
                    _datetime = datetime.combine(_date, _time).astimezone(self.my_std) + HOUR
            _datetime = _datetime.astimezone(tz.UTC)
            if self.last_datetime is not None and _datetime < self.last_datetime:
                # Wurm doesn't have data to identifying date lines for when time hits 24h marker.
                _datetime += timedelta(days=1)
                self.last_datetime = _datetime
            elif _datetime < self.start_datetime:
                self.last_datetime = _datetime
                self.start_line_index = i + 1
            elif _datetime == self.start_datetime:
                self.start_line_index = i
                break
            elif _datetime > self.start_datetime:
                self.start_line_index = i
                break


    def _time_stamp_import(self, line: str, _date: date, _time: time) -> list[date, time]:
        _search = search(r"Logging started ([0-9\-]+)", line)
        if _search is not None:
            _date = date.fromisoformat(_search.group(1))
            return [_date, _time]
        _search = search(r"\[([0-9:]+)\]", line)
        if _search is not None:
            _time = time.fromisoformat(_search.group(1))
            return [_date, _time]
        logger.critical(f'error identify time stamp {line}')
        return [None, None]

    def _reset_line_gen(self) -> None:
        if self.web_file != None:
            self.line_gen = (line for line in self.web_file.text.split(f'\r\n'))
        elif self.file_path != None:
            self.line_gen = (line for line in open(self.file_path, "r"))
        else:
            logger.critical('no valid log data source provided.')
            return

    def process_lines(self):
        _time = self.last_datetime.astimezone(self.tz_org).time()
        _date = self.last_datetime.astimezone(self.tz_org).date()
        esm = None
        self._reset_line_gen()
        _isRarity = False
        rarity_time = None

        for i, line in enumerate(islice(self.line_gen, self.start_line_index, None)):
            if line == "\n":
                continue
            _date, _time = self._time_stamp_import(line, _date, _time)
            _datetime =  datetime.combine(_date, _time).astimezone(self.tz_org).astimezone(tz.UTC)
            if _datetime < self.last_datetime:
                # Wurm doesn't have data to identifying date lines for when time hits 24h marker.
                _datetime += timedelta(days=1)
            self.last_datetime =  _datetime
            if self.end_datetime < _datetime:
                break
            #  Identify and classify the line.
            _search = search(r"Logging started ", line)
            if _search is not None:
                continue
            sm = list(start_messages | where(lambda x:x.line == line[11:].strip()))
            if len(sm) > 0 and esm != None:
                logger.warning(
                    f'Last datetime, {_datetime}. Second start message in, "{line.strip()}". line # {self.start_line_index + i}')
                esm = None
                continue
            if len(sm) > 0:
                esm = EventStartMsg(_datetime, sm[0].action)
                continue
            em = list(end_messages | where(lambda x:x.line == line[11:].strip()))
            if len(em) > 0 and esm == None:
                logger.warning(
                    f'Last datetime, {_datetime}. End message without start message, "{line.strip()}". line # {self.start_line_index + i}')
                esm = None
                continue
            if line[11:].strip() == "You have a moment of inspiration...":
                _isRarity = True
                rarity_time = _datetime
                continue
            if len(em) > 0:
                if rarity_time != None and rarity_time != _datetime:
                    logger.warning(f"rarity time {rarity_time} vrs end time {_datetime} mismatch.")
                action_attempts.append(ActionAttempt(esm.datetime, _datetime, em[0].action, _isRarity))
                esm = None
                rarity_time = None
                _isRarity = False
    
    def get_stagger_durations(self) -> list[(float, int)]:
        ret = list(range(1, len(action_attempts)) 
            | select(lambda x:( 
                    (action_attempts[x].start_time - action_attempts[x-1].start_time).total_seconds(), 
                    x-1)))
        return ret
    
    def get_action_durations(self) -> list[(float, int)]:
        return list(range(0, len(action_attempts)) 
                    | select(lambda x:((action_attempts[x].end_time - action_attempts[x].start_time).total_seconds(),
                    x)))

    def __gen_window(self, interval: int, start: datetime, end: datetime):
        _total = end - start
        _window_cnt = ceil((end - start) / timedelta(seconds=30))
        now = start
        while now < end:
            now += relativedelta(seconds=interval)
            yield now
    
    def check_windows(self, interval: int) -> str:
        g = self.__gen_window(interval, self.start_datetime, self.end_datetime)
        #for w in g:
        length = _window_cnt = ceil(
            (self.end_datetime - self.start_datetime) / timedelta(seconds=interval))
        #a = list(range(length) )

        #a = list (iter(g) select( lambda x: ))
        result = ""
        start = 0
        for dt in g:
            matches = list()
            gen_aa = (a for a in islice(range(len(action_attempts)), start, None, 1))
            for aa in gen_aa:
                if action_attempts[aa].start_time >= dt - relativedelta(seconds=30) and action_attempts[aa].start_time < dt:
                    matches.append(action_attempts[aa])
                if action_attempts[aa].start_time > dt:
                    start = aa
                    break
            r = list(matches | where( lambda x: x.isRarity))
            if not matches:
                result += "W"
            elif matches and r:
                result += "R"
            elif matches and not r:
                result += "X"
        return result
        

if __name__ == "__main__":
    #TODO This whole import section needs improvement. The primary goal is 
    # have it so the code can fetch files from hosts like OneDrive or GoogleDrive. 
    # A users provides direct download link and the code fetches it an does it thing.
    # I also have a way to import a local path if the code was run locally. A command line 
    # tool would be easier to deal with then something hosted on internet.
    embed = '<iframe src="https://onedrive.live.com/embed?cid=8BF93030BCC12554&resid=8BF93030BCC12554%211003&authkey=AFGGQYkSItv2XGM" width="98" height="120" frameborder="0" scrolling="no"></iframe>'
    h_tag = BeautifulSoup(embed, "html5lib")
    web = h_tag.iframe["src"].replace("embed", "download", 1)
    #https://1drv.ms/t/s!AlQlwbwwMPmLh2swsZEmt9SkmupZ?e=ttI16P
    #https://cdn.matix-media.net/dd/36be025d
    start = datetime.fromisoformat('2022-11-07T21:15:40')
    end = datetime.fromisoformat('2022-11-08T06:28:23')
    fil = Path(r'C:\Users\Jason\AppData\Local\Programs\Wurm Online\players\joedobo\logs' + f'\\_Event.2022-{start.month:02}.txt')
    ### END OF IMPORT SECTION. ###
    

    el = EventLog(start, end, file_path=fil)
    total_seconds = el.end_datetime - el.start_datetime
    el.process_lines()
    
    result_str = el.check_windows(30)
    
    stag_dur = np.array(el.get_stagger_durations())
    
    stag_mean = np.mean(stag_dur, axis=0)[0]

    fig = plt.figure(figsize =(10, 7))
    # Creating plot
    plt.boxplot(stag_dur[:,0], False, "")
    # show plot
    plt.show()
    a = 1

    #TODO Trying to use chances per item made is a problem because of variation in times. 
    # And missed window length varies too. I need to focus on looking at the sample 
    # in 30 second intervals. Was at least on action done in it? Where any actions rarity?