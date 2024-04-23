import tensorflow as tf

class WhistleDetectorLoss(object):
    def __init__(self):
        super(WhistleDetectorLoss, self).__init__()
        self.thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
        self.calculated_precision = None
        self.calculated_recall = None
        self.calculated_fscore = None
        self.calculated_accuracy = None
        self.calculated_threshold = None
        self.calculated_combi_score = None

    def precision(self, y_true, y_pred):
        if not tf.is_tensor(self.calculated_precision):
            self.loss(y_true, y_pred)
        return self.calculated_precision

    def recall(self, y_true, y_pred):
        if not tf.is_tensor(self.calculated_recall):
            self.loss(y_true, y_pred)
        return self.calculated_recall

    def fscore(self, y_true, y_pred):
        if not tf.is_tensor(self.calculated_fscore):
            self.loss(y_true, y_pred)
        return self.calculated_fscore

    def accuracy(self, y_true, y_pred):
        if not tf.is_tensor(self.calculated_accuracy):
            self.loss(y_true, y_pred)
        return self.calculated_accuracy

    def threshold(self, y_true, y_pred):
        if not tf.is_tensor(self.calculated_threshold):
            self.loss(y_true, y_pred)
        return self.calculated_threshold
    
    def combi_score(self, y_true, y_pred):
        if not tf.is_tensor(self.calculated_accuracy) or not tf.is_tensor(self.calculated_fscore) or tf.is_tensor(self.calculated_precision):
            self.loss(y_true, y_pred)
        return self.calculated_combi_score

    def whistle_loss(self, y_true, y_pred):
        loss = self.loss(y_true, y_pred)
        return loss

    #@tf.autograph.experimental.do_not_convert
    def loss(self, y_true, y_pred):
        true_confidence = y_true[..., 0]
        pred_confidence = y_pred[..., -1]

        ####################################
        #### F-Score, Recall, Precision ####
        ####################################
        f_beta = 0.5 # > 1 more weight to recall | < 1 more weight to precision
        tps = []
        tns = []
        fps = []
        fns = []
        precisions = []
        recalls = []
        accuracies = []
        f_scores = []
        combi_scores = []
        for t in self.thresholds:
            selected_pred_confidence = tf.cast((pred_confidence > t), 'float')
            tp = tf.cast(true_confidence * selected_pred_confidence, 'float')
            tn = tf.cast((1 - true_confidence) * (1 - selected_pred_confidence), 'float')
            fp = tf.cast((1 - true_confidence) * selected_pred_confidence, 'float')
            fn = tf.cast(true_confidence * (1 - selected_pred_confidence), 'float')

            tps.append(tp)
            tns.append(tn)
            fps.append(fp)
            fns.append(fn)

            tp_sum = tf.reduce_sum(tp, axis=0)
            tn_sum = tf.reduce_sum(tn, axis=0)
            fp_sum = tf.reduce_sum(fp, axis=0)
            fn_sum = tf.reduce_sum(fn, axis=0)

            p = tp_sum / (tp_sum + fp_sum + tf.keras.backend.epsilon())
            r = tp_sum / (tp_sum + fn_sum + tf.keras.backend.epsilon())
            a = (tp_sum + tn_sum)/(tp_sum + tn_sum + fp_sum + fn_sum + tf.keras.backend.epsilon())
            precisions.append(p)
            recalls.append(r)
            accuracies.append(a)

            f_score = ((1 + f_beta ** 2) * tp_sum) / (((1 + f_beta ** 2) * tp_sum) + ((f_beta ** 2) * fn_sum) + fp_sum + tf.keras.backend.epsilon())
            f_score = tf.where(tf.math.is_nan(f_score), tf.zeros_like(f_score), f_score)
            f_scores.append(f_score)

            cs = (tp_sum/(tp_sum + fn_sum + tf.keras.backend.epsilon()) + tn_sum/(tn_sum + fp_sum + tf.keras.backend.epsilon()))/(fp_sum/(fp_sum + fn_sum + tf.keras.backend.epsilon()) + fn_sum/(fp_sum + fn_sum + tf.keras.backend.epsilon()) + tf.keras.backend.epsilon())
            combi_score = cs/(1+tf.abs(cs))
            combi_scores.append(combi_score)

        max_combi_score = tf.argmax(combi_scores)
        self.calculated_precision = tf.gather(precisions, max_combi_score) * 100.
        self.calculated_recall = tf.gather(recalls, max_combi_score) * 100.
        self.calculated_fscore = tf.gather(f_scores, max_combi_score) * 100.
        self.calculated_accuracy = tf.gather(accuracies, max_combi_score) * 100.
        self.calculated_threshold = tf.gather(self.thresholds, max_combi_score)
        self.calculated_combi_score = tf.gather(combi_scores, max_combi_score) * 100.
        # r_w = 0.4
        # a_w = 0.99
        # p_w = 1.15
        # self.calculated_combi_score = ((r_w * self.calculated_recall + a_w * self.calculated_accuracy + p_w * self.calculated_precision) / (r_w + a_w + p_w)) / 100.
        #########################
        #### Confidence Loss ####
        #########################
        loss_confidence = tf.keras.losses.binary_crossentropy(true_confidence, pred_confidence)
        
        return loss_confidence