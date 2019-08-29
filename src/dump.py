def restore_box_from_logits(_delta_yx, _side_distance, valid_indice):
    valid_delta_yx = tf.gather_nd(_delta_yx, valid_indice[..., :-1])
    valid_side_distance = tf.gather_nd(_side_distance, valid_indice[..., :-1])

    # print(valid_indice)
    # print(valid_delta_yx)
    valid_center_yx = tf.cast(valid_indice[..., 1:3], tf.float32) + valid_delta_yx
    valid_center_side = valid_side_distance * (input_h / stride, input_w / stride)

    _boxes = tf.concat(
        [valid_center_yx - valid_center_side[:, ::-1], valid_center_yx + valid_center_side[:, ::-1]], axis=1)
    _boxes *= stride

    # convert to xmin, ymin, xmax, ymax
    _boxes = tf.stack(
        [_boxes[..., 1],
         _boxes[..., 0],
         _boxes[..., 3],
         _boxes[..., 2]],
        axis=1)
    return _boxes


def _iou_loss(output, y_true):
    predict_delta_yx = tf.tanh(output[..., 1:3])
    predict_side_distance = tf.sigmoid(output[..., 3:5])

    true_delta_yx = y_true[..., 1:3]
    true_side_distance = y_true[..., 3:5]

    # print(output)
    # print(y_true)
    # print(tf.where(output[..., 0:1] == 1))
    valid_indice = tf.where(tf.equal(y_true[..., 0:1], 1))
    # valid_indice = tf.keras.backend.print_tensor(valid_indice, message='valid indice')

    pred_box = restore_box_from_logits(predict_delta_yx, predict_side_distance, valid_indice)
    true_box = restore_box_from_logits(true_delta_yx, true_side_distance, valid_indice)

    pred_box_area = (pred_box[..., 2] - pred_box[..., 0]) * (pred_box[..., 3] - pred_box[..., 1])
    true_box_area = (true_box[..., 2] - true_box[..., 0]) * (true_box[..., 3] - true_box[..., 1])

    # x1min, y1min, x1max, y1max, x2min, y2min, x2max, y2max
    # campare_box = tf.concat([pred_box, true_box], axis=1)
    insert_w = tf.maximum(
        tf.minimum(pred_box[..., 2], true_box[..., 2]) - tf.maximum(pred_box[..., 0], true_box[..., 0]), 0)
    insert_h = tf.maximum(
        tf.minimum(pred_box[..., 3], true_box[..., 3]) - tf.maximum(pred_box[..., 1], true_box[..., 1]), 0)

    insert_area = insert_w * insert_h
    # insert_area = tf.Print(insert_area, [insert_area], message='insert area', summarize=100)
    iou = insert_area / (pred_box_area + true_box_area - insert_area)
    # iou = tf.Print(iou, [iou], message='iou: ', summarize=100)

    iou_loss = tf.reduce_sum(-tf.math.log(iou + 1e-7))
    # iou_loss = tf.Print(iou_loss,[iou_loss], message='iou loss: ', summarize=100)
    return iou_loss