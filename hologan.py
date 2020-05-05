import torch
import numpy as np
from base import GAN
from torchvision.utils import save_image

class HoloGAN(GAN):

    def __init__(self, angles, *args, **kwargs):
        super(HoloGAN, self).__init__(*args, **kwargs)
        self.angles = self._angles_to_dict(angles)
        self.rot2idx = {
            'x': 0,
            'y': 1,
            'z': 2
        }

    def _to_radians(self, deg):
        return deg * (np.pi / 180)

    def _angles_to_dict(self, angles):
        angles = {
            'min_angle_x': self._to_radians(angles[0]),
            'max_angle_x': self._to_radians(angles[1]),
            'min_angle_y': self._to_radians(angles[2]),
            'max_angle_y': self._to_radians(angles[3]),
            'min_angle_z': self._to_radians(angles[4]),
            'max_angle_z': self._to_radians(angles[5])
        }
        return angles

    def rot_matrix_x(self, theta):
        """
        theta: measured in radians
        """
        mat = np.zeros((3,3)).astype(np.float32)
        mat[0, 0] = 1.
        mat[1, 1] = np.cos(theta)
        mat[1, 2] = -np.sin(theta)
        mat[2, 1] = np.sin(theta)
        mat[2, 2] = np.cos(theta)
        return mat

    def rot_matrix_y(self, theta):
        """
        theta: measured in radians
        """
        mat = np.zeros((3,3)).astype(np.float32)
        mat[0, 0] = np.cos(theta)
        mat[0, 2] = np.sin(theta)
        mat[1, 1] = 1.
        mat[2, 0] = -np.sin(theta)
        mat[2, 2] = np.cos(theta)
        return mat

    def rot_matrix_z(self, theta):
        """
        theta: measured in radians
        """
        mat = np.zeros((3,3)).astype(np.float32)
        mat[0, 0] = np.cos(theta)
        mat[0, 1] = -np.sin(theta)
        mat[1, 0] = np.sin(theta)
        mat[1, 1] = np.cos(theta)
        mat[2, 2] = 1.
        return mat

    def pad_rotmat(self, theta):
        """theta = (3x3) rotation matrix"""
        return np.hstack((theta, np.zeros((3,1))))

    def sample_angles(self,
                      bs,
                      min_angle_x,
                      max_angle_x,
                      min_angle_y,
                      max_angle_y,
                      min_angle_z,
                      max_angle_z):
        """Sample random yaw, pitch, and roll angles"""
        angles = []
        for i in range(bs):
            rnd_angles = [
                np.random.uniform(min_angle_x, max_angle_x),
                np.random.uniform(min_angle_y, max_angle_y),
                np.random.uniform(min_angle_z, max_angle_z),
            ]
            angles.append(rnd_angles)
        return np.asarray(angles)

    def get_theta(self, angles):
        '''Construct a rotation matrix from angles.

        This uses the Euler angle representation. But
        it should also work if you use an axis-angle
        representation.

        '''
        bs = len(angles)
        theta = np.zeros((bs, 3, 4))

        angles_x = angles[:, 0]
        angles_y = angles[:, 1]
        angles_z = angles[:, 2]
        for i in range(bs):
            theta[i] = self.pad_rotmat(
                np.dot(np.dot(self.rot_matrix_z(angles_z[i]), self.rot_matrix_y(angles_y[i])),
                       self.rot_matrix_x(angles_x[i]))
            )

        return torch.from_numpy(theta).float()

    def prepare_batch(self, batch):
        if len(batch) != 1:
            raise Exception("Expected batch to only contain X")
        X_batch = batch[0].float()
        if self.use_cuda:
            X_batch = X_batch.cuda()
        return [X_batch]

    def sample_z(self, *args, **kwargs):
        return super(HoloGAN, self).sample_z(*args, **kwargs)

    def sample(self, bs, seed=None):
        """Return a sample G(z)"""
        self._eval()
        with torch.no_grad():
            z_batch = self.sample_z(bs, seed=seed)
            angles = self.sample_angles(z_batch.size(0),
                                        **self.angles)
            thetas = self.get_theta(angles)
            if z_batch.is_cuda:
                thetas = thetas.cuda()
            gz = self.g(z_batch, thetas)
        return gz

    def _generate_rotations(self,
                            z_batch,
                            axes=['x', 'y', 'z'],
                            min_angle=None,
                            max_angle=None,
                            num=5):
        dd = dict()
        for rot_mode in axes:
            if min_angle is None:
                min_angle = self.angles['min_angle_%s' % rot_mode]
            if max_angle is None:
                max_angle = self.angles['max_angle_%s' % rot_mode]
            pbuf = []
            with torch.no_grad():
                for p in np.linspace(min_angle, max_angle, num=num):
                    #enc_rot = gan.rotate_random(enc, angle=p)
                    angles = np.zeros((z_batch.size(0), 3)).astype(np.float32)
                    angles[:, self.rot2idx[rot_mode]] += p
                    thetas = self.get_theta(angles)
                    if z_batch.is_cuda:
                        thetas = thetas.cuda()
                    x_fake = self.g(z_batch, thetas)
                    pbuf.append(x_fake*0.5 + 0.5)
            dd[rot_mode] = pbuf
        return dd

    def train_on_instance(self, z, x, **kwargs):
        for key in self.optim:
            self.optim[key].zero_grad()
        self._train()
        losses = {}
        # Train the generator.
        angles = self.sample_angles(z.size(0), **self.angles)
        thetas = self.get_theta(angles)
        angles_t = torch.from_numpy(angles).float().cuda()
        if x.is_cuda:
            thetas = thetas.cuda()
        fake = self.g(z, thetas)
        d_fake, g_z_pred, g_t_pred = self.d(fake)
        gen_loss = self.loss(d_fake, 1)
        g_z_loss = torch.mean((g_z_pred-z)**2)
        g_t_loss = torch.mean((g_t_pred-angles_t)**2)
        if (kwargs['iter']-1) % self.update_g_every == 0:
            (gen_loss + self.lamb*(g_z_loss+g_t_loss)).backward()
            self.optim['g'].step()
        # Train the discriminator.
        self.optim['d'].zero_grad()
        d_fake, d_z_pred, d_t_pred = self.d(fake.detach())
        d_real, _, _ = self.d(x)
        d_loss = self.loss(d_real, 1) + self.loss(d_fake, 0)
        d_z_loss = torch.mean((d_z_pred-z)**2)
        d_t_loss = torch.mean((d_t_pred-angles_t)**2)
        (d_loss + self.lamb*(d_z_loss+d_t_loss)).backward()
        self.optim['d'].step()

        losses['g_loss'] = gen_loss.item()
        losses['d_loss'] = d_loss.item() / 2.
        losses['g_z_loss'] = g_z_loss.item()
        losses['d_z_loss'] = d_z_loss.item()
        losses['g_t_loss'] = g_t_loss.item()
        losses['d_t_loss'] = d_t_loss.item()

        outputs = {
            'x': x.detach(),
            'gz': fake.detach(),
        }
        return losses, outputs
