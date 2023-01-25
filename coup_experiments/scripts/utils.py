from absl import logging

def log_to_file(filename):
  fileh = logging.logging.FileHandler(filename, 'a')
  formatter = logging.logging.Formatter('%(message)s')
  fileh.setFormatter(formatter)

  log = logging.logging.getLogger()
  for h in log.handlers[:]:
    log.removeHandler(h)
  log.addHandler(fileh)

def log_flags(flags, names):
  for name in names:
    logging.info("%s: %s", name, getattr(flags, name))

