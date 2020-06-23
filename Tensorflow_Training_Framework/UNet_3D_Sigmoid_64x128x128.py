#!/usr/bin/env python3
#
# Generate projection data.
#
from tensorflow.keras.layers import Conv3D, UpSampling3D, Concatenate, Input, MaxPooling3D, Activation, BatchNormalization
from tensorflow.keras import Model
import numpy as n

def down_layer(input_layer, filters, kernel_size, pool_downsample, padding = 'same', strides = 1):
    #first conv
    c1 = Conv3D(filters, kernel_size, padding = padding, strides = strides)(input_layer)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    #second conv
    c2 = Conv3D(filters, kernel_size, padding = padding, strides = strides)(c1)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    #max pool downsample
    d1 = MaxPooling3D(pool_size = pool_downsample)(c2)
    return c2, d1
    
def bottle_neck(input_layer, filters, kernel_size, padding = 'same', strides = 1):
    #first conv layer
    b1 = Conv3D(filters, kernel_size, padding = padding, strides = strides)(input_layer)
    b1 = BatchNormalization()(b1)
    b1 = Activation('relu')(b1)
    #second conv layer
    b2 = Conv3D(filters, kernel_size = kernel_size, padding = padding, strides = strides)(b1)
    b2 = BatchNormalization()(b2)
    b2 = Activation('relu')(b2)
    return b2
    
def up_layer(input_layer, skip_layer, filters, upsample_factor, kernel_size, padding = 'same', strides = 1):
    #upsample
    u1 = UpSampling3D(upsample_factor)(input_layer)
    #merge with skip
    m1 = Concatenate()([u1,skip_layer])
    #first conv layer
    c1 = Conv3D(filters, kernel_size, padding = padding, strides = strides)(m1)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    #second conv layer
    c2 = Conv3D(filters, kernel_size, padding = padding, strides = strides)(c1)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    return c2




def UNet(input_shape, kernel_size = 3):
    print('\n')
    #input layer
    i      = Input(input_shape)
    print('input  ', n.shape(i))
    #downsamples
    c1, p1 = down_layer(i , 16 , (3,5,5), (2,2,2))
    print('d1     ', n.shape(c1))
    print('p1     ', n.shape(p1))
    c2, p2 = down_layer(p1, 32 , kernel_size, (2,2,2))
    print('d2     ', n.shape(c2))
    print('p2     ', n.shape(p2))
    c3, p3 = down_layer(p2, 64 , kernel_size, (2,2,2))
    print('d1     ', n.shape(c3))
    print('p3     ', n.shape(p3))
    c4, p4 = down_layer(p3, 128, kernel_size, (2,2,2))
    print('d4     ', n.shape(c4))
    print('p4     ', n.shape(p4))
    c5, p5 = down_layer(p4, 128, (2,3,3), (2,2,2))
    print('d5     ', n.shape(c5))
    print('p5     ', n.shape(p5))
    c6, p6 = down_layer(p5, 128, (2,2,2), (2,2,2))
    print('d6     ', n.shape(c6))
    print('p6     ', n.shape(p6))
    c7, p7 = down_layer(p6, 128, (1,2,2), (1,2,2))
    print('d7     ', n.shape(c7))
    print('p7     ', n.shape(p7))
    #bottle neck
    b      = bottle_neck(p7, 128, kernel_size)
    print('bn     ', n.shape(b))
    #upsamples
    u1     = up_layer(b , c7, 128, (1,2,2),  (1,2,2))
    print('u1     ', n.shape(u1))
    u2     = up_layer(u1 , c6, 128, (2,2,2), (1,2,2))
    print('u2     ', n.shape(u2))
    u3     = up_layer(u2 , c5, 128, (2,2,2), (2,3,3))
    print('u3     ', n.shape(u3))
    u4     = up_layer(u3, c4, 128, (2,2,2), kernel_size)
    print('u4     ', n.shape(u4))
    u5     = up_layer(u4, c3, 64 , (2,2,2), kernel_size)
    print('u5     ', n.shape(u5))
    u6     = up_layer(u5, c2, 32 , (2,2,2), kernel_size)
    print('u6     ', n.shape(u6))
    u7     = up_layer(u6, c1, 16 , (2,2,2), (3,5,5))
    print('u7     ', n.shape(u7))
    #last layer.
    o = Conv3D(1, (1,1,1), padding = 'same', activation = 'sigmoid')(u7)
    print('output ', n.shape(o))
    return Model(inputs = i, outputs = o)



def main(output_channels, dimensions, saveName):
    print('\n Building model...')
    dimZ, dimY, dimX = n.array(dimensions.split(',')).astype(n.int16)
    input_shape      = [dimZ, dimY, dimX, output_channels]
    model            = UNet(input_shape)
    model.save(saveName)
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    print(model.summary())
    print('Model saved to ' + saveName)
    print('\n\n\n\n')
        
        
import argparse
if __name__ == '__main__' :
    usage = 'Written by Cameron Pain. Opens a dicom file.'
    parser = argparse.ArgumentParser(description = usage)
    #positional
    parser.add_argument('output_channels', type = int , help = 'The number of output channels for your model.')
    parser.add_argument('dimensions'     , type = str , help = 'The dimensions of the input images. specify as z,y,x or slice,column, row ')
    parser.add_argument('saveName'       , type = str , help = 'The name of the output .model file.' )
    args = parser.parse_args()
    main(args.output_channels, args.dimensions, args.saveName)


