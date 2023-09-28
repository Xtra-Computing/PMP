import time
import datetime
from torch.utils.tensorboard import SummaryWriter
import os
import json
import yaml

starttime = time.time()

def my_clock():
	'''basically time.clock()
	'''
	return time.time() - starttime
def len_ignore_n(s):
	'''len(s) ignoring \\n
	'''
	s = str(s).strip()
	s = s.replace("\n" , "")

	l = (len(bytes(s , encoding = "utf-8")) - len(s)) // 2 + len(s) #中文
	l += 7 * s.count("\t")											#\t

	return l

def last_len(s):
	'''length of last line
	'''
	s = str(s).strip()
	return len_ignore_n(s.split("\n")[-1])

class Logger:
	'''auto log
	'''
	def __init__(self , mode = [print] , log_path = None , append = ["clock"] , line_length = 90):
		if log_path:
			self.log_fil = open(log_path , "w" , encoding = "utf-8")
		else:
			self.log_fil = None

		self.mode = mode

		if ("write" in mode) and (not log_path):
			raise Exception("Should have a log_path")

		self.append 	= append
		self.line_length = line_length

	def close(self):
		if self.log_fil:
			self.log_fil.close()

	def log(self , content = ""):

		content = self.pre_process(content)

		for x in self.mode:
			if x == "write":
				self.log_fil.write(content + "\n")
				self.log_fil.flush()
			else:
				x(str(content))

	def add_line(self , num = -1 , char = "-"):
		if num < 0:
			num = self.line_length
		self.log(char * num)

	def pre_process(self , content):
		insert_space = self.line_length - last_len(content) #complement to line_length 
		content += " " * insert_space

		for x in self.append: #add something to the end

			y = ""
			if x == "clock":
				y = "%.2fs" % (my_clock())
			elif x == "time":
				y = time.strftime("%Y-%m-%d %H:%M:%S" , time.localtime() )
			else:
				y = x()

			content += "| " + y + " "


		return content
	
class Timer:
    def __init__(self, task_name):
        self.task_name = task_name
        self.tic = 0.0
        self.toc = 0.0

        # 累计计时次数
        self.cnt = 0
        # 总计
        self.total_time = 0.0
        # start与end配对出现
        self.pair_flag = True

    def start(self):
        assert self.pair_flag == True, 'The amount of timer.start() and timer.end() should be the same.'
        self.tic = time.time()
        self.pair_flag = False

    def end(self):
        assert self.pair_flag == False, 'Using timer.start before timer.end'
        self.toc = time.time()
        self.total_time += self.toc - self.tic
        self.cnt += 1
        self.pair_flag = True

    @property
    def avg_time(self):
        return self.total_time / self.cnt


class SummaryBox:
    def __init__(self, task_name, model_name, start_wall_time, log_dir=None, flush_secs=60):
        self.task_name = task_name

        # 这个时间会用作为全局log的时间
        # self.start_wall_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.log_dir = os.path.join('runs', model_name, self.task_name, f"{start_wall_time}") # runs/GAGA/yelp/config.yml
        if log_dir:
            self.log_dir = os.path.join(log_dir, self.log_dir)

        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=flush_secs)

        print(f"[{self.__class__.__name__}] Storing results to {self.log_dir}")

    def update_metrics(self, results, global_step=None):
        fields = results._fields
        for field, value in zip(fields, results):
            tag = f"metrics/{field}"
            self.writer.add_scalar(tag=tag, scalar_value=value, global_step=global_step)

    def update_loss(self, value, mode='train', global_step=None):
        tag = f"loss/{mode}_loss"
        self.writer.add_scalar(tag=tag, scalar_value=value, global_step=global_step)

    def add_figure(self, figure, fig_name, global_step=None):
        tag = f"figures/{fig_name}"
        self.writer.add_figure(tag=tag, figure=figure, global_step=global_step)

    def add_graph(self, model, input_to_model):
        tag = f"graphs/{model.__class__.__name__}"
        self.writer.add_graph(model, input_to_model)

    def save_config(self, configs):
        file_path = os.path.join(self.log_dir, '{}.yml'.format(self.task_name))
        with open(file_path, 'w+') as f:
            yaml.dump(configs, f, sort_keys=True, indent = 2)

    def close(self):
        self.writer.close()
