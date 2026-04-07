from datetime import datetime
from typing import List, Optional, Dict
from .schedules import (
    MONDAY_SCHEDULE,
    TUESDAY_SCHEDULE,
    WEDNESDAY_SCHEDULE,
    THURSDAY_SCHEDULE,
    FRIDAY_SCHEDULE,
    SATURDAY_SCHEDULE,
    SUNDAY_SCHEDULE,
)

class ScheduleContextGenerator:
    SCHEDULES = {
        0: MONDAY_SCHEDULE,
        1: TUESDAY_SCHEDULE,
        2: WEDNESDAY_SCHEDULE,
        3: THURSDAY_SCHEDULE,
        4: FRIDAY_SCHEDULE,
        5: SATURDAY_SCHEDULE,
        6: SUNDAY_SCHEDULE,
    }

    @staticmethod
    def _parse_time_range(time_range: str) -> tuple[datetime.time, datetime.time]:
        start_str, end_str = time_range.split("-")
        start_time = datetime.strptime(start_str, "%H:%M").time()
        end_time = datetime.strptime(end_str, "%H:%M").time()
        return start_time, end_time

        

    @classmethod
    def get_current_activity(cls) -> Optional[str]:
        current_datetime = datetime.now()
        current_time = current_datetime.time()
        current_day = current_datetime.weekday()
        
        schedule = cls.SCHEDULES.get(current_day, {})

        for time_range, activity in schedule.items():
            start_time, end_time = cls._parse_time_range(time_range)
            if start_time > end_time:
                if current_time >= start_time or current_time <= end_time:
                    return activity
            else:
                if start_time <= current_time <= end_time:
                    return activity

        return None

    @classmethod
    def get_schedule_for_day(cls, day: int) -> Dict[str, str]:
        return cls.SCHEDULES.get(day, {})