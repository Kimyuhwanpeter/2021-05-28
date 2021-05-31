# -*- coding:utf-8 -*-
from random import shuffle, random
from F_to_M_model import *

import matplotlib.pyplot as plt
import numpy as np
import easydict
import os

FLAGS = easydict.EasyDict({"img_size": 256,
                           
                           "batch_size": 1,
                           
                           "input_path": "D:/[1]DB/[1]second_paper_DB/AFAD_16_69_DB/backup/fix_AFAD/",

                           "input_txt": "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/race_age_gender_generation/Morph_AFAD_16_63/first_fold/AFAD-F_Morph-M_16_39_40_63/train/female_16_39_train.txt",
                           
                           "ref_path": "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/Morph/All/male_40_63/",
                           
                           "ref_txt": "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/race_age_gender_generation/Morph_AFAD_16_63/first_fold/AFAD-F_Morph-M_16_39_40_63/train/male_40_63_train.txt",
                           
                           "epochs": 200,

                           "lr": 2e-4,
                           
                           "sample_images": "C:/Users/Yuhwan/Downloads/img"})

d_optim = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5, beta_2=0.5)
g_optim = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5, beta_2=0.5)
g_optim_aux = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5, beta_2=0.5)

def input_func(A_path, B_path):

    A_img = tf.io.read_file(A_path)
    B_img = tf.io.read_file(B_path)

    A_img = tf.image.decode_jpeg(A_img, 3)
    B_img = tf.image.decode_jpeg(B_img, 3)

    A_img = tf.image.resize(A_img, [FLAGS.img_size + 15, FLAGS.img_size + 15])
    B_img = tf.image.resize(B_img, [FLAGS.img_size + 15, FLAGS.img_size + 15])
    A_img = tf.image.random_crop(A_img, [FLAGS.img_size, FLAGS.img_size, 3]) / 127.5 - 1.
    B_img = tf.image.random_crop(B_img, [FLAGS.img_size, FLAGS.img_size, 3]) / 127.5 - 1.

    if random() > 0.5:
        A_img = tf.image.flip_left_right(A_img)
        B_img = tf.image.flip_left_right(B_img)

    return A_img, B_img

def encode_style_global_local(model_enc_style_global_local, input, trainable=True):

    N, H, W, C = input.shape

    style_glob, style_loc = input[:, :, :, :C // 2], input[:, :, :, C // 2:]
    style_glob = model_enc_style_global_local(style_glob, trainable)
    style_glob = tf.reduce_mean(tf.reduce_mean(style_glob, 2, keepdims=True), 1, keepdims=True)
    style_glob = tf.tile(style_glob, [N, H, W, 1])

    return tf.concat([style_glob, style_loc], -1)

def encode(model_enc_pre, model_enc, model_enc_style_global_local, A_images, N_cont, trainable=True):

    z_inp_pre = model_enc_pre(A_images, trainable)
    z_inp = model_enc(z_inp_pre, trainable)
    z_inp_C, z_inp_S = z_inp[:, :, :, :N_cont], z_inp[:, :, :, N_cont:]
    z_inp_S = encode_style_global_local(model_enc_style_global_local, z_inp_S)

    return {'cont': z_inp_C, 'middle': z_inp_pre, 'style': z_inp_S}

def decode(model_merge, model_dec, z_input, trainable=True):

    z_inp_combined = model_merge(z_input, trainable)
    rec = model_dec(z_inp_combined, trainable)
    return rec

def decode_aux(model_merge_aux, model_dec_aux, z_input, trainable=True):

    z_inp_combined = model_merge_aux(z_input, trainable)
    rec = model_dec_aux(z_inp_combined, trainable)
    return rec

def reconstruct_image(model_enc_pre, model_enc, model_enc_style_global_local, input, N_cont,
                      model_merge, model_dec, zero_cont = False, zero_style = False):   # 이미지 생성할때 쓰임
    inp = input

    z_inp = encode(model_enc_pre, model_enc, model_enc_style_global_local, input, N_cont)
    z_new = tf.concat([z_inp['cont'], z_inp['style']], -1)

    if zero_cont:
        z_new[:, :, :, :N_cont] = torch.zeros_like(z_new[:, :, :, :N_cont])
            
    if zero_style:
        z_new[:, :, :, N_cont:] = torch.zeros_like(z_new[:, :, :, N_cont:])

    rec = decode(model_merge, model_dec, z_new)

    return rec

def stylize_image(model_enc_pre_inp, model_enc_inp, model_enc_style_global_local_inp, 
                  model_enc_pre_dst, model_enc_dst, model_enc_style_global_local_dst,
                  input, N_cont,
                  model_peer_reg, model_merge, model_dec, trainable=True):

    inp, tar = input

    z_inp = encode(model_enc_pre_inp, model_enc_inp, model_enc_style_global_local_inp, inp, N_cont)
    z_tar = encode(model_enc_pre_dst, model_enc_dst, model_enc_style_global_local_dst, tar, N_cont)

    z_new = model_peer_reg([z_inp['cont'], z_inp['style'], z_tar['cont'], z_tar['style']], trainable)
    z_new_S = z_new[:, :, :, N_cont:]

    rec = decode(model_merge, model_dec, z_new)

    return rec, z_inp['cont'], z_inp['style'], z_tar['cont'], z_tar['style']

def define_G(model_enc_pre_inp, model_enc_inp, model_enc_style_global_local_inp,
             model_enc_pre_dst, model_enc_dst, model_enc_style_global_local_dst,
             model_enc_pre_rec, model_enc_rec, model_enc_style_global_local_rec,
             model_merge_o, model_dec_o,
             model_merge_rec_id, model_dec_rec_id, model_peer_reg,
             model_enc_pre_rec_id, model_enc_rec_id, model_enc_style_global_local_rec_id,
             model_merge_rec_style_id, model_dec_rec_style_id,
             model_enc_pre_rec_style_id, model_enc_rec_style_id, model_enc_style_global_local_rec_style_id,
             model_merge_aux_rec_id_aux, model_dec_aux_rec_id_aux,
             model_enc_pre_rec_id_aux, model_enc_rec_id_aux, model_enc_style_global_local_rec_id_aux,
             model_merge_aux_rec_style_id_aux, model_dec_aux_rec_style_id_aux,
             model_enc_pre_rec_style_id_aux, model_enc_rec_style_id_aux, model_enc_style_global_local_rec_style_id_aux,
             A_images, B_images, N_cont, trainable=True):
    feat = {}
    feat['inp_src'] = encode(model_enc_pre_inp, model_enc_inp, model_enc_style_global_local_inp, A_images, N_cont, trainable)   # A

    feat['inp_tar'] = encode(model_enc_pre_inp, model_enc_inp, model_enc_style_global_local_inp, B_images, N_cont, trainable)   # B
    feat['dst_src'] = encode(model_enc_pre_dst, model_enc_dst, model_enc_style_global_local_dst, B_images, N_cont, trainable)   # A style-B

    feat['dst_tar'] = encode(model_enc_pre_dst, model_enc_dst, model_enc_style_global_local_dst, A_images, N_cont, trainable)   # B style-A

    # peer regul
    z_cont_new = model_peer_reg([feat['inp_src']['cont'], feat['inp_src']['style'],
                                 feat['dst_tar']['cont'], feat['dst_tar']['style']], trainable)
    feat['peer_reg'] = {'cont':z_cont_new[:, :, :, :N_cont], 'middle':None, 'style':z_cont_new[:, :, :, N_cont:]}

    # merge and decode
    rec = decode(model_merge_o, model_dec_o, z_cont_new, trainable)

    # Identity mapping
    feat['rec'] = encode(model_enc_pre_rec, model_enc_rec, model_enc_style_global_local_inp, rec, N_cont)
    z_cont_new_id = tf.concat([feat['inp_src']['cont'], feat['inp_src']['style']], -1)
    rec_id = decode(model_merge_rec_id, model_dec_rec_id, z_cont_new_id)
    feat['rec_id'] = encode(model_enc_pre_rec_id, model_enc_rec_id, model_enc_style_global_local_dst, rec_id, N_cont)

    z_style_new_id = tf.concat([feat['dst_tar']['cont'], feat['dst_tar']['style']], -1)
    rec_style_id = decode(model_merge_rec_style_id, model_dec_rec_style_id, z_style_new_id)
    #feat['rec_style_id'] = encode(model_enc_pre_rec_style_id, model_enc_rec_style_id, model_enc_style_global_local_rec_style_id, rec_style_id, N_cont)

    # Auxiliary decoder
    z_cont_new_id_aux = tf.concat([feat['inp_src']['cont'], feat['inp_src']['style']], -1)
    rec_id_aux = decode_aux(model_merge_aux_rec_id_aux, model_dec_aux_rec_id_aux, z_cont_new_id_aux)
    feat['rec_id_aux'] = encode(model_enc_pre_rec_id_aux, model_enc_rec_id_aux, model_enc_style_global_local_dst, rec_id_aux, N_cont)

    # Auxiliary decoder processing style
    z_style_new_id_aux = tf.concat([feat['dst_tar']['cont'], feat['dst_tar']['style']], -1)
    rec_style_id_aux = decode_aux(model_merge_aux_rec_style_id_aux, model_dec_aux_rec_style_id_aux, z_style_new_id_aux)
    feat['rec_style_id_aux'] = encode(model_enc_pre_rec_style_id_aux, model_enc_rec_style_id_aux, model_enc_style_global_local_inp, rec_style_id_aux, N_cont)

    return rec, rec_id, rec_style_id, rec_id_aux, rec_style_id_aux, feat

def define_D(model_discirm, input):
    logits = model_discirm(input, True)
    return logits

def smooth_L1_loss(y_true, y_pred, beta=1.0):
    x = tf.abs(y_true - y_pred)
    x = tf.where(x < beta, (0.5 * (x)**2) / beta, 0.5 * (x - 0.5 * beta))
    return x

def cal_loss(model_enc_pre_inp, model_enc_inp, model_enc_style_global_local_inp,
             model_enc_pre_dst, model_enc_dst, model_enc_style_global_local_dst,
             model_enc_pre_rec, model_enc_rec, model_enc_style_global_local_rec,
             model_merge_o, model_dec_o,
             model_merge_rec_id, model_dec_rec_id, model_peer_reg,
             model_enc_pre_rec_id, model_enc_rec_id, model_enc_style_global_local_rec_id,
             model_merge_rec_style_id, model_dec_rec_style_id,
             model_enc_pre_rec_style_id, model_enc_rec_style_id, model_enc_style_global_local_rec_style_id,
             model_merge_aux_rec_id_aux, model_dec_aux_rec_id_aux,
             model_enc_pre_rec_id_aux, model_enc_rec_id_aux, model_enc_style_global_local_rec_id_aux,
             model_merge_aux_rec_style_id_aux, model_dec_aux_rec_style_id_aux,
             model_enc_pre_rec_style_id_aux, model_enc_rec_style_id_aux, model_enc_style_global_local_rec_style_id_aux,
             A_images, B_images, N_cont, model_discirm_A, model_discirm_B, model_discirm_rec_A, model_discirm_fake_B):

    with tf.GradientTape(persistent=True) as d_tape, tf.GradientTape(persistent=True) as g_tape, tf.GradientTape(persistent=True) as g_aux_tape:

        fake_B, rec_A, rec_style_B, rec_A_aux, rec_style_B_aux, feat = \
            define_G(model_enc_pre_inp, model_enc_inp, model_enc_style_global_local_inp,
                        model_enc_pre_dst, model_enc_dst, model_enc_style_global_local_dst,
                        model_enc_pre_rec, model_enc_rec, model_enc_style_global_local_rec,
                        model_merge_o, model_dec_o,
                        model_merge_rec_id, model_dec_rec_id, model_peer_reg,
                        model_enc_pre_rec_id, model_enc_rec_id, model_enc_style_global_local_rec_id,
                        model_merge_rec_style_id, model_dec_rec_style_id,
                        model_enc_pre_rec_style_id, model_enc_rec_style_id, model_enc_style_global_local_rec_style_id,
                        model_merge_aux_rec_id_aux, model_dec_aux_rec_id_aux,
                        model_enc_pre_rec_id_aux, model_enc_rec_id_aux, model_enc_style_global_local_rec_id_aux,
                        model_merge_aux_rec_style_id_aux, model_dec_aux_rec_style_id_aux,
                        model_enc_pre_rec_style_id_aux, model_enc_rec_style_id_aux, model_enc_style_global_local_rec_style_id_aux,
                        A_images, B_images, N_cont)
        realA_logits = define_D(model_discirm_A, [A_images, B_images])
        fake_A_logits = define_D(model_discirm_rec_A, [rec_A, A_images])

        realB_logits = define_D(model_discirm_B, [B_images, A_images])
        fake_B_logits = define_D(model_discirm_fake_B, [fake_B, A_images])

        D_loss = (tf.reduce_mean((realA_logits - tf.ones_like(realA_logits))**2) \
            + tf.reduce_mean((fake_A_logits - tf.zeros_like(fake_A_logits))**2)) / 2 \
            + (tf.reduce_mean((realB_logits - tf.ones_like(realB_logits))**2) \
            + tf.reduce_mean((fake_B_logits - tf.zeros_like(fake_B_logits))**2)) / 2

        G_loss = (tf.reduce_mean((fake_A_logits - tf.ones_like(fake_A_logits))**2) \
            + tf.reduce_mean((fake_B_logits - tf.ones_like(fake_B_logits))**2)) / 2

        cycle_loss = tf.reduce_mean((rec_A - A_images)**2) * 25.0
        cycle_loss += tf.reduce_mean((rec_style_B - B_images)**2) * 25.0
        cycle_loss /= 2.0

        post_dists1 = tf.reduce_mean(tf.reduce_sum(smooth_L1_loss(feat['rec']['cont'], feat['inp_src']['cont']), -1))
        post_dists2 = tf.reduce_mean(tf.reduce_sum(smooth_L1_loss(feat['rec']['style'], feat['dst_tar']['style']), -1))
        loss_trans = (post_dists1 + post_dists2) / 2.0

        G_loss = cycle_loss + G_loss + loss_trans
        
        ##################################################################################################################
        # content latent loss
        post_dists1 = smooth_L1_loss(feat['rec']['cont'], feat['inp_src']['cont'])  # 1 64 64 256
        post_dists1 = tf.reshape(post_dists1, [post_dists1.shape[0], post_dists1.shape[3], -1]) # 1 256, 64*64
        post_dists1 = tf.reduce_mean(tf.reduce_sum(post_dists1, 1), 1)  # 1, 1
        post_dists2 = smooth_L1_loss(feat['rec_id']['cont'], feat['inp_src']['cont'])
        post_dists2 = tf.reshape(post_dists2, [post_dists2.shape[0], post_dists2.shape[3], -1]) # 1 256, 64*64
        post_dists2 = tf.reduce_mean(tf.reduce_sum(post_dists2, 1), 1)  # 1, 1
        loss_z_cont = tf.reduce_mean(tf.concat([post_dists1, post_dists2], 0))

        # style latent loss
        loss_z_style = 0.0
        dst_tgt_sty = feat['dst_tar']['style']  # B style -A
        inp_tar_sty = feat['inp_tar']['style']  # B
        dst_src_sty = feat['dst_src']['style']  # A style -B
        inp_src_sty = feat['inp_src']['style']  # A
        pos_dists1 = smooth_L1_loss(dst_tgt_sty,inp_tar_sty) # B style과 B의 유사도?
        pos_dists1 = tf.reshape(pos_dists1, [pos_dists1.shape[0], pos_dists1.shape[3], -1])
        pos_dists1 = tf.reduce_mean(tf.reduce_sum(pos_dists1, 1), 1)
        pos_dists2 = smooth_L1_loss(dst_src_sty,inp_src_sty) # A style 과 A의 유사도?
        pos_dists2 = tf.reshape(pos_dists2, [pos_dists2.shape[0], pos_dists2.shape[3], -1])
        pos_dists2 = tf.reduce_mean(tf.reduce_sum(pos_dists2, 1), 1)
        pos_dists = tf.reduce_mean(tf.concat([pos_dists1, pos_dists2], 0))

        neg_dist1 = smooth_L1_loss(dst_src_sty, inp_tar_sty)    # A style 과 B의 유사도?
        neg_dist1 = tf.reshape(neg_dist1, [neg_dist1.shape[0], neg_dist1.shape[3], -1])
        neg_dist1 = tf.reduce_mean(tf.reduce_sum(neg_dist1, 1), 1)
        neg_dist2 = smooth_L1_loss(dst_tgt_sty, inp_src_sty)    # B style 과 A의 유사도?
        neg_dist2 = tf.reshape(neg_dist2, [neg_dist2.shape[0], neg_dist2.shape[3], -1])
        neg_dist2 = tf.reduce_mean(tf.reduce_sum(neg_dist2, 1), 1)
        neg_dist = tf.reduce_mean(tf.concat([neg_dist1, neg_dist2], 0))
        
        loss_z_style += (pos_dists + tf.maximum(tf.constant(0.0, tf.float32), 1.0 - neg_dist)) * 1.0

        # cycle aux loss
        cycle_loss_aux = 0.0
        p = smooth_L1_loss(feat['rec_id_aux']['cont'], feat['inp_src']['cont'])
        p = tf.reduce_mean(tf.reduce_sum(tf.reshape(p, [p.shape[0], p.shape[3], -1]), 1))
        cycle_loss_aux += p
        p = smooth_L1_loss(feat['rec_id_aux']['style'],feat['inp_src']['style'])
        p = tf.reduce_mean(tf.reduce_sum(tf.reshape(p, [p.shape[0], p.shape[3], -1]), 1))
        cycle_loss_aux += p
        p = smooth_L1_loss(feat['rec_style_id_aux']['style'], feat['dst_tar']['style'])
        p = tf.reduce_mean(tf.reduce_sum(tf.reshape(p, [p.shape[0], p.shape[3], -1]), 1))
        cycle_loss_aux += p
        p = smooth_L1_loss(feat['rec_id_aux']['middle'], feat['inp_src']['middle'])
        p = tf.reduce_mean(tf.reduce_sum(tf.reshape(p, [p.shape[0], p.shape[3], -1]), 1))
        cycle_loss_aux += p
        p = smooth_L1_loss(feat['rec_style_id_aux']['middle'] , feat['dst_tar']['middle'])
        p = tf.reduce_mean(tf.reduce_sum(tf.reshape(p, [p.shape[0], p.shape[3], -1]), 1))
        cycle_loss_aux += p
        cycle_loss_aux /= 5.0

        # id loss
        id_loss_aux = tf.reduce_mean((rec_A_aux - A_images)**2)
        id_loss_aux += tf.reduce_mean((rec_style_B_aux - B_images)**2)

        # combine loss
        G_loss_aux = loss_z_cont + loss_z_style + cycle_loss_aux + id_loss_aux

    d_grads = d_tape.gradient(D_loss, model_discirm_A.trainable_variables \
                                    + model_discirm_B.trainable_variables \
                                    + model_discirm_rec_A.trainable_variables \
                                    + model_discirm_fake_B.trainable_variables)

    g_grads = g_tape.gradient(G_loss, model_enc_inp.trainable_variables \
                                    + model_enc_dst.trainable_variables \
                                    + model_enc_style_global_local_inp.trainable_variables \
                                    + model_enc_style_global_local_dst.trainable_variables \
                                    + model_enc_pre_inp.trainable_variables \
                                     + model_enc_pre_dst.trainable_variables \
                                    + model_merge_rec_id.trainable_variables \
                                    + model_dec_rec_id.trainable_variables \
                                    + model_merge_rec_style_id.trainable_variables \
                                    + model_dec_rec_style_id.trainable_variables \
                                    + model_enc_pre_rec.trainable_variables \
                                    + model_enc_rec.trainable_variables)
    g_grads2 = g_tape.gradient(G_loss, model_peer_reg.trainable_variables)
    g_grads3 = g_tape.gradient(G_loss, model_dec_o.trainable_variables \
                                     + model_merge_o.trainable_variables)

    g_grads_aux = g_aux_tape.gradient(G_loss_aux, model_enc_pre_rec.trainable_variables \
                                                + model_enc_rec.trainable_variables \
                                                #+ model_enc_style_global_local_rec.trainable_variables \
                                                + model_enc_pre_rec_id.trainable_variables \
                                                + model_enc_rec_id.trainable_variables \
                                                #+ model_enc_style_global_local_rec_id.trainable_variables \
                                                + model_merge_aux_rec_id_aux.trainable_variables \
                                                + model_dec_aux_rec_id_aux.trainable_variables \
                                                + model_enc_pre_rec_id_aux.trainable_variables \
                                                + model_enc_rec_id_aux.trainable_variables \
                                                #+ model_enc_style_global_local_rec_id_aux.trainable_variables \
                                                + model_merge_aux_rec_style_id_aux.trainable_variables \
                                                + model_dec_aux_rec_style_id_aux.trainable_variables \
                                                + model_enc_pre_rec_style_id_aux.trainable_variables \
                                                + model_enc_rec_style_id_aux.trainable_variables)
                                                #+ model_enc_style_global_local_rec_style_id_aux.trainable_variables)

    d_optim.apply_gradients(zip(d_grads, model_discirm_A.trainable_variables \
                                     + model_discirm_B.trainable_variables \
                                     + model_discirm_rec_A.trainable_variables \
                                     + model_discirm_fake_B.trainable_variables))
    g_optim.apply_gradients(zip(g_grads, model_enc_inp.trainable_variables \
                                    + model_enc_dst.trainable_variables \
                                    + model_enc_style_global_local_inp.trainable_variables \
                                    + model_enc_style_global_local_dst.trainable_variables \
                                    + model_enc_pre_inp.trainable_variables \
                                     + model_enc_pre_dst.trainable_variables \
                                    + model_merge_rec_id.trainable_variables \
                                    + model_dec_rec_id.trainable_variables \
                                    + model_merge_rec_style_id.trainable_variables \
                                    + model_dec_rec_style_id.trainable_variables \
                                    + model_enc_pre_rec.trainable_variables \
                                    + model_enc_rec.trainable_variables))
    g_optim.apply_gradients(zip(g_grads2, model_peer_reg.trainable_variables))
    g_optim.apply_gradients(zip(g_grads3, model_dec_o.trainable_variables \
                                     + model_merge_o.trainable_variables))

    g_optim_aux.apply_gradients(zip(g_grads_aux, model_enc_pre_rec.trainable_variables \
                                                + model_enc_rec.trainable_variables \
                                                #+ model_enc_style_global_local_rec.trainable_variables \
                                                + model_enc_pre_rec_id.trainable_variables \
                                                + model_enc_rec_id.trainable_variables \
                                                #+ model_enc_style_global_local_rec_id.trainable_variables \
                                                + model_merge_aux_rec_id_aux.trainable_variables \
                                                + model_dec_aux_rec_id_aux.trainable_variables \
                                                + model_enc_pre_rec_id_aux.trainable_variables \
                                                + model_enc_rec_id_aux.trainable_variables \
                                                #+ model_enc_style_global_local_rec_id_aux.trainable_variables \
                                                + model_merge_aux_rec_style_id_aux.trainable_variables \
                                                + model_dec_aux_rec_style_id_aux.trainable_variables \
                                                + model_enc_pre_rec_style_id_aux.trainable_variables \
                                                + model_enc_rec_style_id_aux.trainable_variables))
                                                #+ model_enc_style_global_local_rec_style_id_aux.trainable_variables))


    return D_loss, G_loss, G_loss_aux

def main():

    model_enc_inp, model_enc_pre_inp, N_cont, N_style, mul = Encoder_multiscale(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    model_enc_rec, model_enc_pre_rec, N_cont, N_style, mul = Encoder_multiscale(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    model_enc_dst, model_enc_pre_dst, N_cont, N_style, mul = Encoder_multiscale(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    model_enc_rec_id, model_enc_pre_rec_id, N_cont, N_style, mul = Encoder_multiscale(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    model_enc_rec_style_id, model_enc_pre_rec_style_id, N_cont, N_style, mul = Encoder_multiscale(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    model_enc_rec_id_aux, model_enc_pre_rec_id_aux, N_cont, N_style, mul = Encoder_multiscale(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    model_enc_rec_style_id_aux, model_enc_pre_rec_style_id_aux, N_cont, N_style, mul = Encoder_multiscale(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))

    model_enc_style_global_local_inp = Global_Encoder(input_shape=(64, 64, 256), N_style=N_style)
    model_enc_style_global_local_rec = Global_Encoder(input_shape=(64, 64, 256), N_style=N_style)
    model_enc_style_global_local_rec_id = Global_Encoder(input_shape=(64, 64, 256), N_style=N_style)
    model_enc_style_global_local_rec_style_id = Global_Encoder(input_shape=(64, 64, 256), N_style=N_style)
    model_enc_style_global_local_rec_style_id_aux = Global_Encoder(input_shape=(64, 64, 256), N_style=N_style)
    model_enc_style_global_local_rec_id_aux = Global_Encoder(input_shape=(64, 64, 256), N_style=N_style)
    model_enc_style_global_local_dst = Global_Encoder(input_shape=(64, 64, 256), N_style=N_style)
    
    model_merge_rec_id = Decoder_merge(input_shape=(64, 64, 768))
    model_merge_rec_style_id = Decoder_merge(input_shape=(64, 64, 768))
    model_merge_aux_rec_id_aux = Decoder_merge(input_shape=(64, 64, 768))
    model_merge_aux_rec_style_id_aux = Decoder_merge(input_shape=(64, 64, 768))
    model_merge_o = Decoder_merge(input_shape=(64, 64, 768))

    model_dec_rec_style_id = Decoder_model(input_shape=(64, 64, 64*4))
    model_dec_aux_rec_id_aux = Decoder_model(input_shape=(64, 64, 64*4))
    model_dec_rec_id = Decoder_model(input_shape=(64, 64, 64*4))
    model_dec_aux_rec_style_id_aux = Decoder_model(input_shape=(64, 64, 64*4))
    model_dec_o = Decoder_model(input_shape=(64, 64, 64*4))

    model_peer_reg = PeerRegularizationLayerAtt_model()

    model_discirm_A = Discriminator(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    model_discirm_rec_A = Discriminator(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    model_discirm_B = Discriminator(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    model_discirm_fake_B = Discriminator(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    

    input_data = np.loadtxt(FLAGS.input_txt, dtype="<U100", usecols=0, skiprows=0)
    input_data = [FLAGS.input_path + img for img in input_data]

    ref_data = np.loadtxt(FLAGS.ref_txt, dtype="<U100", usecols=0, skiprows=0)
    ref_data = [FLAGS.ref_path + img for img in ref_data]

    count = 0
    for epoch in range(FLAGS.epochs):
        A = list(zip(input_data, ref_data))
        shuffle(A)
        input_data, ref_data = zip(*A)
    
        input_data, ref_data = np.array(input_data), np.array(ref_data)

        TR_gener = tf.data.Dataset.from_tensor_slices((input_data, ref_data))
        TR_gener = TR_gener.shuffle(len(input_data))
        TR_gener = TR_gener.map(input_func)
        TR_gener = TR_gener.batch(FLAGS.batch_size)
        TR_gener = TR_gener.prefetch(tf.data.experimental.AUTOTUNE)

        tr_iter = iter(TR_gener)
        tr_idx = len(input_data) // FLAGS.batch_size
        for step in range(tr_idx):

            A_images, B_images = next(tr_iter)
            

            D_loss, G_loss, G_loss_aux = cal_loss(model_enc_pre_inp, model_enc_inp, model_enc_style_global_local_inp,
                                                  model_enc_pre_dst, model_enc_dst, model_enc_style_global_local_dst,
                                                  model_enc_pre_rec, model_enc_rec, model_enc_style_global_local_rec,
                                                  model_merge_o, model_dec_o,
                                                  model_merge_rec_id, model_dec_rec_id, model_peer_reg,
                                                  model_enc_pre_rec_id, model_enc_rec_id, model_enc_style_global_local_rec_id,
                                                  model_merge_rec_style_id, model_dec_rec_style_id,
                                                  model_enc_pre_rec_style_id, model_enc_rec_style_id, model_enc_style_global_local_rec_style_id,
                                                  model_merge_aux_rec_id_aux, model_dec_aux_rec_id_aux,
                                                  model_enc_pre_rec_id_aux, model_enc_rec_id_aux, model_enc_style_global_local_rec_id_aux,
                                                  model_merge_aux_rec_style_id_aux, model_dec_aux_rec_style_id_aux,
                                                  model_enc_pre_rec_style_id_aux, model_enc_rec_style_id_aux, model_enc_style_global_local_rec_style_id_aux,
                                                  A_images, B_images, N_cont, model_discirm_A, model_discirm_B, model_discirm_rec_A, model_discirm_fake_B)
            print("================================")
            print("Epoch = {} [{}/{}]\nD loss = {}\nG loss = {}\nG aux loss = {}".format(epoch, step + 1, tr_idx, D_loss, G_loss, G_loss_aux))
            print("================================")

            if count % 100 ==0:
                input = [A_images, B_images]
                fake_B, z_cont_real_A, z_style_real_A, z_cont_style_B, z_style_B = stylize_image(model_enc_pre_inp, model_enc_inp, model_enc_style_global_local_inp,
                                                                                                 model_enc_pre_dst, model_enc_dst, model_enc_style_global_local_dst,
                                                                                                 input, N_cont,
                                                                                                 model_peer_reg, model_merge_o, model_dec_o, trainable=False)
                #feat['inp_src'] = encode(model_enc_pre_inp, model_enc_inp, model_enc_style_global_local_inp, A_images, N_cont, False)
                #feat['dst_tar'] = encode(model_enc_pre_dst, model_enc_dst, model_enc_style_global_local_dst, B_images, N_cont, False)

                # peer regul
                #z_cont_new = model_peer_reg([feat['inp_src']['cont'], feat['inp_src']['style'],
                #                                feat['dst_tar']['cont'], feat['dst_tar']['style']], False)
                #feat['peer_reg'] = {'cont':z_cont_new[:, :, :, :N_cont], 'middle':None, 'style':z_cont_new[:, :, :, N_cont:]}

                # merge and decode
                #rec = decode(model_merge_o, model_dec_o, z_cont_new, False)

                plt.imsave(FLAGS.sample_images + "/generated_img_{}.jpg".format(count), fake_B[0].numpy() * 0.5 + 0.5)
                plt.imsave(FLAGS.sample_images + "/A_img_{}.jpg".format(count), A_images[0].numpy() * 0.5 + 0.5)
                plt.imsave(FLAGS.sample_images + "/B_img_{}.jpg".format(count), B_images[0].numpy() * 0.5 + 0.5)


            count += 1


if __name__ == "__main__":
    main()
