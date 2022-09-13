from dataclasses import dataclass
from datetime import datetime, date, time, timedelta
from typing import List, Tuple, Type
from typing_extensions import Self
from pathlib import Path, PurePath
from io import SEEK_END
from asyncio import run, create_task, sleep
from re import search
from itertools import islice
from json import load as jsonload
from pipe import select, where
from time import sleep
from collections import namedtuple
import logging
import logging.config
import os
import yaml
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_LEVEL = logging.INFO
def log_setup(log_cfg_path='cfg.yaml'):
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


def search_timestamp(line: str) -> time:
    _search = search(r"\[([0-9:]+)\]", line)
    if _search is not None:
        return time.fromisoformat(_search.group(1))
    else:
        return None

def search_datestamp(line: str) -> date:
    _search = search(r"Logging started ([0-9\-]+)", line)
    if _search is not None:
        return date.fromisoformat(_search.group(1))
    else:
        return None


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


EventStartMsg = namedtuple("EventStartMsg", "datetime action")

class EventLog():

    def __init__(self, log_path: Path, start_datetime: datetime, end_datetime: datetime):
        self.log_path = log_path
        self.tell_pos = None
        self.last_modified = None

        self.start_datetime = start_datetime
        self.end_datetime = end_datetime
        if not isinstance(self.log_path, PurePath):
            logger.CRITICAL('combat log file path is not a Path object')
            return
        if not self.log_path.exists() or not self.log_path.is_file():
            logger.CRITICAL(
                'file exists..{},  file is a file..{}, {}'.format(self.log_path.exists(), 
                self.log_path.is_file(), str(self.log_path)))
            return
        self.last_modified = datetime.fromtimestamp(self.log_path.stat().st_mtime)
        
        # Find line index that has a time stamp which is at most equal to the start date.
        self.start_line_index = 0
        _time = _date = None
        with open(self.log_path) as f:
            for line in f:
                self.start_line_index += 1
                _date1 = search_datestamp(line)
                if _date1 != None:
                    _date = _date1
                    continue
                _time1 = search_timestamp(line)
                if _time1 == None:
                    logger.CRITICAL('Time does not have valid time.')
                    return
                _time = _time1
                _datetime = datetime.combine(_date, _time)
                if _datetime < self.start_datetime:
                    self.last_datetime = _datetime
                if _datetime == self.start_datetime:
                    self.start_line_index -= 1
                    break
                elif _datetime > self.start_datetime:
                    self.start_line_index -= 1
                    break

    def log_changes(self):
        while self.end_datetime > self.last_datetime:
            self.process_lines()

    async def updating_log_changes(self):
        while self.end_datetime > self.last_datetime:
            while datetime.fromtimestamp(self.log_path.stat().st_mtime) == self.last_modified:
                await sleep(0.1)
            self.process_lines()

    def process_lines(self):
        _time = self.last_datetime.time()
        _date = self.last_datetime.date()
        _datetime = datetime.combine(_date, _time)
        esm = None
        with open(self.log_path) as f:
            line_cnt = 0
            for line in islice(f, self.start_line_index, None):
                # Get time and date from lines in log.
                # The date and time identifiers are on different lines. And date only appears on log in.
                # Logging started 2022-09-01
                # [00:08:05] The shards doesn't fit.
                line_cnt += 1
                _date1 = search_datestamp(line)
                if _date1 != None:
                    _date = _date1
                    continue
                _time1 = search_timestamp(line)
                if _time1 == None:
                    # With the exception of line, "Logging started", their should always be a time stamp.
                    logger.error(
                        f'Last datetime, {_datetime}. Invalid time stamp in, "{line.strip()}". line # {self.start_line_index + line_cnt}')
                    return
                _time = _time1
                _datetime1 = datetime.combine(_date, _time)
                if _datetime1 < _datetime:
                    # Wurm doesn't have date identifying lines for when time hits 24h marker.
                    _date += timedelta(days=1)
                    _datetime1 = datetime.combine(_date, _time)
                else:
                    _datetime = _datetime1
                if self.end_datetime < _datetime:
                    break
                #  Identify and classify the line.
                sm = list(start_messages | where(lambda x:x.line == line[11:].strip()))
                if len(sm) > 0 and esm != None:
                    logger.warning(
                        f'Last datetime, {_datetime}. Second start message in, "{line.strip()}". line # {self.start_line_index + line_cnt}')
                    esm = None
                    continue
                if len(sm) > 0:
                    esm = EventStartMsg(_datetime, sm[0].action)
                    continue
                em = list(end_messages | where(lambda x:x.line == line[11:].strip()))
                if len(em) > 0 and esm == None:
                    logger.warning(
                        f'Last datetime, {_datetime}. End message without start message, "{line.strip()}". line # {self.start_line_index + line_cnt}')
                    esm = None
                    continue
                if len(em) > 0:
                    action_attempts.append(ActionAttempt(esm.datetime, _datetime, em[0].action))
                    esm = None
    
    def get_out_of_window(self, window_time: int) -> list:
        for ix in range(1, len(action_attempts)):
            compare_items(action_attempts[ix], action_attempts[ix - 1])
    
    def get_stagger_durations(self) -> list[float]:
        ret = list(range(1, len(action_attempts)) 
            | select(lambda x: action_attempts[x].start_time - action_attempts[x-1].start_time)
            | select(lambda x: x.total_seconds()))
        return ret
    
    def get_action_durations(self) -> list[float]:
        return list(action_attempts 
                    | select(lambda x:(x.end_time - x.start_time).total_seconds()))
                   

if __name__ == "__main__":
    el = EventLog(Path 
        (r'C:\Users\Jason\AppData\Local\Programs\Wurm Online\players\joedobo\logs\_Event.2022-09.txt'),
            datetime.fromisoformat('2022-09-02T18:42:00'),
            datetime.fromisoformat('2022-09-03T00:56:00'))
    el.process_lines()
    act_dur = el.get_action_durations()
    stag_dur = el.get_stagger_durations()

    #TODO The duration lists aren't helpful explaining why a data pint is what it is. 
    # I need a way to go look at outrageous data points and see what happened.
    # Maybe a named tuple with the data point and a indices references for its
    # cause from action_attempts.

    fig = plt.figure(figsize =(10, 7))
    # Creating plot
    plt.boxplot(stag_dur)
    # show plot
    plt.show()
    a = 1