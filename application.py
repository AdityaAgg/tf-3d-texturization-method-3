import tensorflow as tf
from opt import ConstrainedOpt
from model import sgan
import dataset as dataset

# initialize
tf.reset_default_graph()


config_proto = config = tf.ConfigProto()
config_proto.gpu_options.allow_growth = True
sess = tf.Session(config=config_proto)
model_object = sgan.Model(1)
dataset = dataset.Dataset()

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(model_object.vars_G + model_object.vars_E)
saver.restore(sess, "params/sgan_model.ckpt")

# train
#model_object.train_model(sess, dataset, 25)

# test

model_object.generate_one_sample(dataset, sess)


# saver


#saver.save(sess, 'params_b/sgan')

# opt_engine = ConstrainedOpt(model)

# initialize application
# app = QApplication(sys.argv)
# app.setStyleSheet(qdarkstyle.load_stylesheet(pyside=False))
# window = MainWindow(opt_engine)
# window.setWindowTitle("pix2vox")
# window.show()
# window.viewerWidget.interactor.Initialize()
# sys.exit(app.exec_())
