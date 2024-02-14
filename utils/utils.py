"""
    MIT License
    Copyright (c) 2024 Reza NasiriGerdeh

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

import os
import numpy as np

import logging
logger = logging.getLogger("utils")


class ResultFile:
    def __init__(self, result_file_name, mode):
        # create root directory for results
        result_root = './results'
        if not os.path.exists(result_root):
            os.mkdir(result_root)

        # open result file
        self.result_file = open(file=f'{result_root}/{result_file_name}', mode=mode)

    def write_header(self, header):
        self.result_file.write(f'{header}\n')
        self.result_file.flush()

    def write_result(self, epoch, result_list):
        digits_precision = 8

        result_str = f'{epoch},'
        for result in result_list:
            if result != '-':
                result = np.round(result, digits_precision)
            result_str += f'{result},'

        # remove final comma
        result_str = result_str[0:-1]

        self.result_file.write(f'{result_str}\n')
        self.result_file.flush()

    def close(self):
        self.result_file.close()

