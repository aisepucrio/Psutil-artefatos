from codecarbon import EmissionsTracker
import psutil
import csv
import os
from collections.abc import Container
import tensorflow as tf
from tqdm import tqdm

tfk = tf.keras
tfkl = tf.keras.layers

class DownShift(tfkl.Layer):
    def __init__(self, name='down_shift'):
        super(DownShift, self).__init__(name=name)
        self.pad = tfkl.ZeroPadding2D(((1,0),(0,0)))
        self.crop = tfkl.Cropping2D(((0,1),(0,0)))

    def call(self, x):
        return self.pad(self.crop(x))

class RightShift(tfkl.Layer):
    def __init__(self, name='right_shift'):
        super(RightShift, self).__init__(name=name)
        self.pad = tfkl.ZeroPadding2D(((0,0),(1,0)))
        self.crop = tfkl.Cropping2D(((0,0),(0,1)))

    def call(self, x):
        return self.pad(self.crop(x))

class DownShiftedConv(tfkl.Layer):
    def __init__(self, filters, kernel_size=(2, 3), strides=1, name='down_shifted_conv'):
        super(DownShiftedConv, self).__init__(name=name)
        self.padding = tfkl.ZeroPadding2D(
            padding=(
                (kernel_size[0] - 1, 0),
                ((kernel_size[1] - 1) // 2, (kernel_size[1] - 1) // 2)
            )
        )
        self.conv = tfkl.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='valid'
        )

    def call(self, x):
        return self.conv(self.padding(x))

class DownShiftedConvTranspose(tfkl.Layer):
    def __init__(self, filters, kernel_size=(2, 3), strides=1, name='down_shifted_conv_transpose'):
        super(DownShiftedConvTranspose, self).__init__(name=name)
        self.conv = tfkl.Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            output_padding=strides - 1,
            strides=strides,
            padding='valid'
        )
        self.crop = tfkl.Cropping2D(
            cropping=(
                (0, kernel_size[0] - 1),
                ((kernel_size[1] - 1) // 2, (kernel_size[1] - 1) // 2)
            )
        )

    def call(self, x):
        return self.crop(self.conv(x))

class DownRightShiftedConv(tfkl.Layer):
    def __init__(self, filters, kernel_size=(2, 2), strides=1, name='downright_shifted_conv'):
        super(DownRightShiftedConv, self).__init__(name=name)
        self.padding = tfkl.ZeroPadding2D(
            padding=(
                (kernel_size[0] - 1, 0),
                (kernel_size[1] - 1, 0)
            )
        )
        self.conv = tfkl.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='valid'
        )

    def call(self, x):
        return self.conv(self.padding(x))

class DownRightShiftedConvTranspose(tfkl.Layer):
    def __init__(self, filters, kernel_size=(2, 2), strides=1, name='downright_shifted_conv_transpose'):
        super(DownRightShiftedConvTranspose, self).__init__(name=name)
        self.conv = tfkl.Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            output_padding=strides - 1,
            padding='valid'
        )
        self.crop = tfkl.Cropping2D(
            cropping=(
                (0, kernel_size[0] - 1),
                (0, kernel_size[1] - 1)
            )
        )

    def call(self, x):
        return self.crop(self.conv(x))

class ResidualBlock(tfkl.Layer):
    def __init__(self, dropout_rate=0.2, name='pixelcnn_layer'):
        super(ResidualBlock, self).__init__(name=name)
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        input_shape, _, context_shape = input_shape
        hidden_dim = input_shape[-1]
        self.dropout = tfkl.Dropout(rate=self.dropout_rate, name='dropout')
        self.v_conv = DownShiftedConv(filters=hidden_dim, name='v_conv')
        self.h_conv = DownRightShiftedConv(filters=hidden_dim, name='h_conv')
        self.v_conv_2 = DownShiftedConv(filters=2 * hidden_dim, name='v_conv_2')
        self.h_conv_2 = DownRightShiftedConv(filters=2 * hidden_dim, name='h_conv_2')
        self.skip_conv = tfkl.Conv2D(filters=hidden_dim, kernel_size=1, name='skip_conv')
        if len(context_shape) > 0:
            self.has_context = True
            self.context_v = tfkl.Dense(units=2 * hidden_dim, name='context_v')
            self.context_h = tfkl.Dense(units=2 * hidden_dim, name='context_h')
        else:
            self.has_context = False

    def call(self, inputs, training=False):
        v_stack, h_stack, context = inputs
        hidden_v = self.v_conv(tf.nn.relu(v_stack))
        hidden_h = self.h_conv(tf.nn.relu(h_stack))
        hidden_h += self.skip_conv(tf.nn.relu(hidden_v))
        hidden_h = self.dropout(hidden_h, training=training)
        hidden_v = self.dropout(hidden_v, training=training)
        hidden_v = self.v_conv_2(tf.nn.relu(hidden_v))
        hidden_h = self.h_conv_2(tf.nn.relu(hidden_h))
        if self.has_context:
            hidden_h += self.context_h(context)[:, None, None, :]
            hidden_v += self.context_v(context)[:, None, None, :]
        h, sigmoid_h = tf.split(hidden_h, num_or_size_splits=2, axis=-1)
        v, sigmoid_v = tf.split(hidden_v, num_or_size_splits=2, axis=-1)
        hidden_h = h * tf.math.sigmoid(sigmoid_h)
        hidden_v = v * tf.math.sigmoid(sigmoid_v)
        hidden_h += h_stack
        hidden_v += v_stack
        return hidden_v, hidden_h

class PixelCNNplus(tfk.Model):
    def __init__(self, hidden_dim, dropout_rate=0.2, n_res=5,
                 n_downsampling=2, n_mix=5, name='pixelcnn_pp'):
        super(PixelCNNplus, self).__init__(name=name)
        self.n_res = n_res
        self.hidden_dim = hidden_dim
        self.n_mix = n_mix
        self.n_downsampling = n_downsampling
        self.dropout_rate = dropout_rate

    def unpack_inputs(self, inputs):
        if isinstance(inputs, Container):
            x, context = inputs
        else:
            x, context = inputs, tf.constant(0.)
        return x, context

    def _build_first_convolutions(self):
        self.first_conv_v = DownShiftedConv(
            kernel_size=(2, 3),
            filters=self.hidden_dim,
            name='first_conv_v'
        )
        self.first_conv_h_v = DownRightShiftedConv(
            kernel_size=(1, 3),
            filters=self.hidden_dim,
            name='first_conv_h_v'
        )
        self.first_conv_h_h = DownShiftedConv(
            kernel_size=(2, 1),
            filters=self.hidden_dim,
            name='first_conv_h_h'
        )

    def _build_downsampling_blocks(self):
        self.downsampling_res_blocks = [
            [
                ResidualBlock(
                    dropout_rate=self.dropout_rate,
                    name=f'ds_res_block{i}_{j}'
                ) for j in range(self.n_res)
            ]
            for i in range(self.n_downsampling + 1)
        ]
        self.downsampling_convs_v = [
            DownShiftedConv(
                filters=self.hidden_dim,
                strides=2,
                name=f'downsampling_conv_v_{i}'
            )
            for i in range(self.n_downsampling)
        ]
        self.downsampling_convs_h = [
            DownRightShiftedConv(
                filters=self.hidden_dim,
                strides=2,
                name=f'downsampling_conv_h_{i}'
            )
            for i in range(self.n_downsampling)
        ]

    def _build_upsampling_blocks(self):
        self.upsampling_res_blocks = [
            [
                ResidualBlock(
                    dropout_rate=self.dropout_rate,
                    name=f'us_res_block{i}_{j}'
                ) for j in range(self.n_res)
            ]
            for i in range(self.n_downsampling + 1)
        ]
        self.upsampling_convs_v = [
            DownShiftedConvTranspose(
                filters=self.hidden_dim,
                strides=2,
                name=f'upsampling_conv_v_{i}'
            )
            for i in range(self.n_downsampling)
        ]
        self.upsampling_convs_h = [
            DownRightShiftedConvTranspose(
                filters=self.hidden_dim,
                strides=2,
                name=f'upsampling_conv_h_{i}'
            )
            for i in range(self.n_downsampling)
        ]

    def _build_residual_connections(self):
        n_res_connections = (self.n_downsampling + 1) * (self.n_res + 1)
        self.res_conv_h = [
            tfkl.Conv2D(
                filters=self.hidden_dim,
                kernel_size=1,
                name=f'res_conv_h_{i}'
            )
            for i in range(n_res_connections)
        ]
        self.res_conv_v = [
            tfkl.Conv2D(
                filters=self.hidden_dim,
                kernel_size=1,
                name=f'res_conv_v_{i}'
            )
            for i in range(n_res_connections)
        ]

    def _build_final_convolutions(self):
        self.final_conv_v = tfkl.Conv2D(
            filters=self.n_mix * self.n_component_per_mix,
            kernel_size=1,
            name='final_conv_v'
        )
        self.final_conv_h = tfkl.Conv2D(
            filters=self.n_mix * self.n_component_per_mix,
            kernel_size=1,
            name='final_conv_h'
        )
        self.final_conv = tfkl.Conv2D(
            filters=self.n_mix * self.n_component_per_mix,
            kernel_size=1,
            name='final_conv'
        )

    def build(self, input_shape):
        x, _ = self.unpack_inputs(input_shape)
        self.image_shape = input_shape[1:]
        n_channels = input_shape[-1]
        if n_channels == 1:
            self.n_component_per_mix = 3
        elif n_channels == 3:
            self.n_component_per_mix = 10
        self.down_shift = DownShift()
        self.right_shift = RightShift()
        self._build_first_convolutions()
        self._build_downsampling_blocks()
        self._build_upsampling_blocks()
        self._build_residual_connections()
        self._build_final_convolutions()

    def call(self, inputs, training=False):
        x, context = self.unpack_inputs(inputs)
        v_stack = self.down_shift(self.first_conv_v(x))
        h_stack = self.down_shift(self.first_conv_h_h(x)) + \
                  self.right_shift(self.first_conv_h_v(x))
        residuals_h, residuals_v = [h_stack], [v_stack]
        for ds in range(self.n_downsampling + 1):
            for res_block in self.downsampling_res_blocks[ds]:
                v_stack, h_stack = res_block((v_stack, h_stack, context), training)
                residuals_h.append(h_stack)
                residuals_v.append(v_stack)
            if ds < self.n_downsampling:
                v_stack = self.downsampling_convs_v[ds](tf.nn.relu(v_stack))
                h_stack = self.downsampling_convs_h[ds](tf.nn.relu(h_stack))
                residuals_h.append(h_stack)
                residuals_v.append(v_stack)
        residuals_v = [
            res_conv_v(tf.nn.relu(res_v))
            for res_conv_v, res_v in zip(self.res_conv_v, residuals_v)
        ]
        residuals_h = [
            res_conv_h(tf.nn.relu(res_h))
            for res_conv_h, res_h in zip(self.res_conv_h, residuals_h)
        ]
        v_stack = residuals_v.pop()
        h_stack = residuals_h.pop()
        for us in range(self.n_downsampling + 1):
            for res_block in self.upsampling_res_blocks[us]:
                v_stack, h_stack = res_block((v_stack, h_stack, context), training)
                v_stack += residuals_v.pop()
                h_stack += residuals_h.pop()
            if us < self.n_downsampling:
                v_stack = self.upsampling_convs_v[us](tf.nn.relu(v_stack))
                h_stack = self.upsampling_convs_h[us](tf.nn.relu(h_stack))
                v_stack += residuals_v.pop()
                h_stack += residuals_h.pop()
        outputs = self.final_conv_h(tf.nn.relu(h_stack)) + \
                  self.final_conv_v(tf.nn.relu(v_stack))
        outputs = self.final_conv(tf.nn.relu(outputs))
        return outputs

    def sample(self, n, context=tf.constant(0.)):
        if tf.rank(context) == 1:
            context = tf.tile(context[None, :], [n, 1])
        height, width, channels = self.image_shape
        n_pixels = height * width
        samples = tf.random.uniform(
            (n, height, width, channels), minval=-1. + 1e-5, maxval=1. - 1e-5)
        for pos in tqdm(range(n_pixels), desc="Sampling PixelCNN++"):
            h = pos // height
            w = pos % height
            logits = self((samples, context))[:, h, w, :]
            if channels == 1:
                pi, mu, logvar = tf.split(logits, num_or_size_splits=3, axis=-1)
            else:
                (pi, mu_r, mu_g, mu_b, logvar_r, logvar_g, logvar_b, alpha,
                 beta, gamma) = tf.split(logits, num_or_size_splits=10, axis=-1)
                alpha = tf.math.tanh(alpha)
                beta = tf.math.tanh(beta)
                gamma = tf.math.tanh(gamma)
                mu = tf.stack([mu_r, mu_g, mu_b], axis=2)
                logvar = tf.stack([logvar_r, logvar_g, logvar_b], axis=2)
            logvar = tf.maximum(logvar, -7.)
            components = tf.random.categorical(logits=pi, num_samples=1)
            mu = tf.gather(mu, components, axis=1, batch_dims=1)
            logvar = tf.gather(logvar, components, axis=1, batch_dims=1)
            u = tf.random.uniform(tf.shape(mu), minval=1e-5, maxval=1. - 1e-5)
            x = mu + tf.exp(logvar) * (tf.math.log(u) - tf.math.log(1. - u))
            if channels == 3:
                alpha = tf.gather(alpha, components, axis=1, batch_dims=1)
                beta = tf.gather(beta, components, axis=1, batch_dims=1)
                gamma = tf.gather(gamma, components, axis=1, batch_dims=1)
                x_r = x[:, 0, 0]
                x_g = x[:, 0, 1] + alpha[:, 0] * x_r
                x_b = x[:, 0, 2] + beta[:, 0] * x_r + gamma[:, 0] * x_g
                x = tf.stack([x_r, x_g, x_b], axis=-1)
            updates = tf.clip_by_value(x, -1., 1.)
            indices = tf.constant([[i, h, w] for i in range(n)])
            samples = tf.tensor_scatter_nd_update(samples, indices, updates)
        return samples

def discretized_logistic_mix_loss(y_true, y_pred):
    _, H, W, C = y_true.shape
    num_pixels = float(H * W * C)
    if C == 1:
        pi, mu, logvar = tf.split(y_pred, num_or_size_splits=3, axis=-1)
        mu = tf.expand_dims(mu, axis=3)
        logvar = tf.expand_dims(logvar, axis=3)
    else:
        (pi, mu_r, mu_g, mu_b, logvar_r, logvar_g, logvar_b, alpha,
         beta, gamma) = tf.split(y_pred, num_or_size_splits=10, axis=-1)
        alpha = tf.math.tanh(alpha)
        beta = tf.math.tanh(beta)
        gamma = tf.math.tanh(gamma)
        red = y_true[:,:,:,0:1]
        green = y_true[:,:,:,1:2]
        mu_g = mu_g + alpha * red
        mu_b = mu_b + beta * red + gamma * green
        mu = tf.stack([mu_r, mu_g, mu_b], axis=3)
        logvar = tf.stack([logvar_r, logvar_g, logvar_b], axis=3)
    logvar = tf.maximum(logvar, -7.)
    y_true = tf.expand_dims(y_true, axis=-1)
    def cdf(x):
        return tf.nn.sigmoid((x - mu) * tf.exp(-logvar))
    def log_cdf(x):
        return tf.math.log_sigmoid((x - mu) * tf.exp(-logvar))
    def log_one_minus_cdf(x):
        return -tf.math.softplus((x - mu) * tf.exp(-logvar))
    def log_pdf(x):
        norm = (x - mu) * tf.exp(-logvar)
        return norm - logvar - 2. * tf.math.softplus(norm)
    half_pixel = 1 / 255.
    cdf_plus = cdf(y_true + half_pixel)
    cdf_min = cdf(y_true - half_pixel)
    log_cdf_plus = log_cdf(y_true + half_pixel)
    log_one_minus_cdf_min = log_one_minus_cdf(y_true - half_pixel)
    cdf_delta = cdf_plus - cdf_min
    cdf_delta = tf.maximum(cdf_delta, 1e-12)
    approx_log_cdf_delta = log_pdf(y_true) - tf.math.log(127.5)
    log_probs = tf.where(cdf_delta > 1e-5, tf.math.log(cdf_delta), approx_log_cdf_delta)
    log_probs = tf.where(y_true > 0.999, log_one_minus_cdf_min, log_probs)
    log_probs = tf.where(y_true < -0.999, log_cdf_plus, log_probs)
    log_probs = tf.reduce_sum(log_probs, axis=3)
    log_probs += tf.nn.log_softmax(pi)
    log_probs = tf.math.reduce_logsumexp(log_probs, axis=-1)
    log_probs = tf.reduce_sum(log_probs, axis=[1, 2])
    bits_per_dim = -log_probs / num_pixels / tf.math.log(2.)
    return bits_per_dim

tracker = EmissionsTracker(project_name=r"C:\Users\PUC\Documents\AISE\ecosustain-replication-study\artefatos\tf2-published-models\pixelcnn_plus\model.py")
tracker.start()
mem_start = psutil.virtual_memory().used / (1024**2)
cpu_start = psutil.cpu_percent(interval=None)
mem_end = psutil.virtual_memory().used / (1024**2)
cpu_end = psutil.cpu_percent(interval=None)
csv_file = "psutil_data.csv"
file_exists = os.path.isfile(csv_file)
with open(csv_file, "a", newline="") as csvfile:
    writer = csv.writer(csvfile)
    if not file_exists:
        writer.writerow(["file", "mem_start_MB", "mem_end_MB", "mem_diff_MB", "cpu_start_percent", "cpu_end_percent"])
    writer.writerow([
        __file__,
        f"{mem_start:.2f}",
        f"{mem_end:.2f}",
        f"{mem_end - mem_start:.2f}",
        f"{cpu_start:.2f}",
        f"{cpu_end:.2f}"
    ])
tracker.stop()
