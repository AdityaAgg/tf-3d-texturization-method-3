import tensorflow as tf
from opt import ConstrainedOpt
from model import sgan
import dataset as dataset

# initialize
tf.reset_default_graph()
config_proto = config = tf.ConfigProto()
config_proto.gpu_options.allow_growth = True
sess = tf.Session(config=config_proto)
model_object = sgan.Model()
dataset = dataset.Dataset()

sess.run(tf.global_variables_initializer())
model_object.print_something()
# train
model_object.train_model(sess, dataset, 10)



# saver
t_vars = tf.train.Saver(model.vars_G + model.vars_E + model.vars_D)
saver = tf.train.Saver(t_vars)
saver.save(sess, 'params_b/sgan')

# opt_engine = ConstrainedOpt(model)

# initialize application
# app = QApplication(sys.argv)
# app.setStyleSheet(qdarkstyle.load_stylesheet(pyside=False))
# window = MainWindow(opt_engine)
# window.setWindowTitle("pix2vox")
# window.show()
# window.viewerWidget.interactor.Initialize()
# sys.exit(app.exec_())
