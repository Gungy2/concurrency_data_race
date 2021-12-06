from enum import Enum, auto
from typing import List, Union
from parse import compile
import numpy as np
from itertools import chain
from prettytable import PrettyTable
import copy


class CommandType(Enum):
    acq = auto()
    rel = auto()
    rd = auto()
    wr = auto()

    def __str__(self):
        return str(self.name)


class DataRaceException(Exception):
    pass


class Command:
    p = compile("{command}({thread:d},{id})")

    def __init__(self, input: str) -> None:
        result = Command.p.parse(input)
        if result is None:
            raise Exception(f"Invalid command: {input}")
        self.type = CommandType[result["command"].strip().lower()]
        self.thread = result["thread"]
        self.id = result["id"].strip()

    def __str__(self) -> str:
        return f"{self.type}({self.thread},{self.id})"


class VectorClock:
    def __init__(self, length: int, index: Union[int, None] = None) -> None:
        self.vector = np.zeros(length, dtype=np.uint32)
        if index is not None:
            self.vector[index] = 1

    def union(self, other: "VectorClock"):
        np.maximum(self.vector, other.vector, self.vector)

    def update(self, index: int, value: int):
        self.vector[index] = value  #

    def get(self, index: int) -> int:
        return self.vector[index]

    def __copy__(self) -> "VectorClock":
        new = VectorClock(0)
        new.vector = np.copy(self.vector)
        return new

    def __le__(self, other: "VectorClock") -> bool:
        return (self.vector < other.vector).all()

    def __ge__(self, other: "VectorClock") -> bool:
        return (self.vector > other.vector).all()

    def __str__(self) -> str:
        return "(" + ", ".join(map(str, self.vector)) + ")"


class Program:
    def __init__(self, no_threads: int, locks: List[str], vars: List[str]) -> None:
        self.no_threads = no_threads
        self.c = [VectorClock(no_threads, i) for i in range(no_threads)]
        self.l = {lock: VectorClock(no_threads) for lock in locks}
        self.r = {var: VectorClock(no_threads) for var in vars}
        self.w = {var: VectorClock(no_threads) for var in vars}

        self.table = PrettyTable()
        self.table.field_names = (
            ["Command"]
            + [f"C{i}" for i in range(no_threads)]
            + [f"L{lock}" for lock in locks]
            + list(chain.from_iterable((f"R{var}", f"W{var}") for var in vars))
        )

        self.__add_row()

    def execute_command(self, command: Command):
        if command.type == CommandType.acq:
            self.c[command.thread].union(self.l[command.id])

        elif command.type == CommandType.rd:
            if self.w[command.id] <= self.c[command.thread]:
                raise DataRaceException(f"Write-read data race detected at: {command}")
            self.r[command.id][command.thread] = self.c[command.thread][command.thread]

        elif command.type == CommandType.wr:
            if self.w[command.id] <= self.c[command.thread]:
                raise DataRaceException(f"Write-write data race detected at: {command}")
            if self.r[command.id] <= self.c[command.thread]:
                raise DataRaceException(f"Read-write data race detected at: {command}")
            self.w[command.id].update(
                command.thread, self.c[command.thread].get(command.thread)
            )

        elif command.type == CommandType.rel:
            self.l[command.id] = np.copy(self.c[command.thread])
            self.c[command.thread][command.thread] += 1

        else:
            raise ValueError("Invalid command")
        self.__add_row(command)

    def __add_row(self, command: Union[Command, None] = None):
        self.table.add_row(
            ["Initial" if command is None else command]
            + [copy.copy(self.c[i]) for i in range(self.no_threads)]
            + [copy.copy(self.l[lock]) for lock in self.l]
            + list(
                chain.from_iterable(
                    (copy.copy(self.r[var]), copy.copy(self.w[var])) for var in self.r
                )
            )
        )

    def print_table(self):
        print(self.table)


if __name__ == "__main__":
    program = Program(2, ["m", "n"], ["x", "y"])
    program.execute_command(Command("acq(1, m)"))
    program.execute_command(Command("wr (1, x)"))
    program.print_table()
