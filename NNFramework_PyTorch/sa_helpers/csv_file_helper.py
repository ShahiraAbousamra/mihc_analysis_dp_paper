import csv

class CSV_Helper:
    def __init__(self):
        self.filepath = None;
        self.newline = None;
        self.delimiter = None;
        self.file = None;
        self.writer = None;

    def create_file(self, filepath, newline='', delimiter=','):
        self.filepath = filepath;
        self.newline = newline;
        self.delimiter = delimiter;
        self.file = open(self.filepath, 'w', newline=self.newline);
        self.writer = csv.writer(self.file, delimiter=self.delimiter);

    def write(self, row):
        if(self.writer is None):
            return False;
        self.writer.writerow(row);  
        return True;

    def close_file(self):
        if(self.file is None):
            return;
        self.file.close();  

    def open_read_file(self, filepath, newline='\r\n', delimiter=','):
        self.filepath = filepath;
        self.newline = newline;
        self.delimiter = delimiter;
        self.file = open(self.filepath, 'r', newline=self.newline);
        self.reader = csv.reader(self.file, delimiter=self.delimiter);
        self.reader_indx = 0;
        self.reader_size = len(self.reader);
        
    def read_rows(self, n=1):
        if(self.reader is None):
            return None;
        if(self.reader_indx >= self.reader_size):
            return None;
        start_indx = self.reader_indx;
        end_indx = self.reader_indx + n;
        if(end_indx > self.reader_size):
            end_indx = self.reader_size;
        rows = self.reader[start_indx: end_indx];
        self.reader_indx = end_indx;  
        return rows;
