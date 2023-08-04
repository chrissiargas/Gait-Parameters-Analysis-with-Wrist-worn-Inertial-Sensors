from tensorboard import program
model = 'Romi'
tracking_address = './logs' + '/' + model + '_model_tb'  # the path of your log file. (accEncoderTb, fullModelTb)

if __name__ == "__main__":
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    tb.main()


