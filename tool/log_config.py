import logging
import time
import os

def log_config(name='logging'):
	if not os.path.exists('./log'):
		os.mkdir('./log')
	logdatetime = time.strftime("%Y-%m-%d-%H:%M:%S--")
	logger = logging.getLogger()
	logger.setLevel(logging.INFO)


	formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

	fh = logging.FileHandler('./log/' + logdatetime + name + '.log')
	fh.setLevel(logging.INFO)
	fh.setFormatter(formatter)
	logger.addHandler(fh)

	ch = logging.StreamHandler()
	ch.setLevel(logging.INFO)
	ch.setFormatter(formatter)
	logger.addHandler(ch)

	return