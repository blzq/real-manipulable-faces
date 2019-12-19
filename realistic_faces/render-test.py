import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import dirt
import dirt.lighting
import dirt.matrices
import dirt.projection

DATA_PATH = os.path.realpath(os.path.join(__file__, '../../data'))
SHAPE_BASIS_PATH = os.path.join(DATA_PATH, "shape_basis.npy")
SHAPE_MEAN_PATH = os.path.join(DATA_PATH, "shape_mean.npy")
EXPR_BASIS_PATH = os.path.join(DATA_PATH, "expression_basis.npy")
COLOR_BASIS_PATH = os.path.join(DATA_PATH, "color_basis.npy")
MESH_FACES_PATH = os.path.join(DATA_PATH, "mesh_faces.npy")
TEXTURE_PATH = os.path.join(DATA_PATH, 'rainbow.png')

if __name__ == '__main__':
    canvas_width, canvas_height = 512, 512

    shape_mean = np.load(SHAPE_MEAN_PATH).astype(np.float32)
    shape_basis = np.load(SHAPE_BASIS_PATH).astype(np.float32)
    expr_basis = np.load(EXPR_BASIS_PATH).astype(np.float32)
    faces = np.load(MESH_FACES_PATH).T
    texture = tf.cast(tf.image.decode_png(tf.io.read_file(TEXTURE_PATH)), tf.float32) / 255
    uv_map = texture[:, :, :2] * 512
    uv_map = tf.image.resize(uv_map, [canvas_height, canvas_width])

    shape_params = np.random.standard_normal([shape_basis.shape[1]]).astype(np.float32)
    expr_params = np.random.standard_normal([expr_basis.shape[1]]).astype(np.float32)
    shape_params = shape_params * 10.
    expr_params = expr_params * 10.

    verts_offsets = np.matmul(shape_basis, shape_params) + np.matmul(expr_basis, expr_params)

    raw_verts = np.reshape(shape_mean, [-1, 3]) + np.reshape(verts_offsets, [-1, 3])

    normals = dirt.lighting.vertex_normals(raw_verts, faces)

    # Homogenous coordinates
    verts_obj = tf.concat([
        raw_verts,
        tf.ones_like(raw_verts[:, -1:])
    ], axis=1)

    view_matrix = dirt.matrices.compose(
        dirt.matrices.rodrigues([0., 0., 0.]),
        dirt.matrices.translation([0., 0., -5000]))

    verts_camera = tf.matmul(verts_obj, view_matrix)
    projection_matrix = dirt.matrices.perspective_projection(
        near=50.0, far=10000., right=1.0, aspect=1.)
    verts_clip = tf.matmul(verts_camera, projection_matrix)

    def shader_fn(gbuffer, texture, light_direction, uv_map):

        # Unpack the different attributes from the G-buffer
        mask = gbuffer[:, :, :1]
        normals = gbuffer[:, :, 1:4]
        uvs = tf.cast(gbuffer[:, :, 4:], tf.int32)

        # Sample the texture at locations corresponding to each pixel
        # this defines the unlit material color at each point
        unlit_colors = tf.gather_nd(texture, uvs)

        # Calculate a simple grey ambient lighting component
        ambient_contribution = unlit_colors * [0.4, 0.4, 0.4]

        # Calculate a diffuse (Lambertian) lighting component
        diffuse_contribution = dirt.lighting.diffuse_directional(
            tf.reshape(normals, [-1, 3]),
            tf.reshape(unlit_colors, [-1, 3]),
            light_direction, light_color=[0.6, 0.6, 0.6], double_sided=True
        )
        diffuse_contribution = tf.reshape(diffuse_contribution, [canvas_height, canvas_width, 3])

        pixels = (diffuse_contribution + ambient_contribution) * mask + [0., 0., 0.3] * (1. - mask)

        return pixels

    max_x = tf.reduce_max(tf.abs(verts_obj[:, 0]))
    max_y = tf.reduce_max(tf.abs(verts_obj[:, 1]))
    max_xy = np.array([max_x, max_y]).reshape([1, 2])
    fixed_indices = tf.cast(verts_obj[:, :2] / max_xy * 256 + 256, tf.int32)
    indices = tf.gather_nd(uv_map, fixed_indices[:, :, ::-1])  # xy to yx

    light_direction = tf.linalg.l2_normalize([1., -0.3, -0.5])
    pixels = dirt.rasterise_deferred(
        vertices=verts_clip,
        vertex_attributes=tf.concat([
            tf.ones_like(verts_clip[:, :1]),  # mask
            normals,                          # normals
            indices                           # uv coordinates
        ], axis=1),
        faces=faces,
        background_attributes=tf.zeros([canvas_height, canvas_width, 6]),
        shader_fn=shader_fn,
        shader_additional_inputs=[texture, light_direction, uv_map]
    )

    plt.imshow(pixels)
    plt.show()
