"""Offline synthetic checks: no instrument hardware or services required."""

from pathlib import Path
import tempfile
import unittest

from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import numpy as onp

from heimdallrdr.fringe import (
    BASELINE_MATRIX, _gaussian_plane, _window, extract_phasors, phasors_to_pd,
    fit_splodge_centres, extract_pd_from_fits, plot_splodge_fits, SplodgeFitError,
    telescope_pd_welch, ExtractionResult,
)
from heimdallrdr._fits import index_fits, read_cube_batches, utc64
from heimdallrdr.DataBundler import DataBundler


CENTRES = onp.array([[5.2, 1.3], [1.4, 7.3], [9.3, 5.4],
                    [-5.4, 6.2], [-9.3, 1.7], [4.3, -8.2]])


def fringe_frames(n=20, size=32, noise=0.05):
    rng = onp.random.default_rng(40)
    y, x = onp.indices((size, size)) - size//2
    frames = onp.zeros((n, size, size))
    phases = rng.uniform(-onp.pi, onp.pi, (n, 6))
    envelope = onp.exp(-(x*x+y*y)/(2*(size/7)**2))
    for b, (u, v) in enumerate(CENTRES):
        frames += 20*envelope*onp.cos(2*onp.pi*(u*x+v*y)/size+phases[:, b, None, None])
    return frames + rng.normal(0, noise, frames.shape)


class CentreTests(unittest.TestCase):
    def test_fractional_ellipse_plane_and_central_mask(self):
        size = 48
        seeds = onp.array([[5., 0], [0, 10], [12, 5], [-10, 8], [-12, -8], [4, -13]])
        truth = seeds + [.27, -.31]
        v, u = onp.indices((size, size))
        u, v = (u+size//2)%size-size//2, (v+size//2)%size-size//2
        power = 10 + .02*u + .03*v
        # A central peak that would pull the baseline-12 fit without masking.
        power = power + 1e6*onp.exp(-(u*u+v*v)/.4)
        for cx, cy in truth:
            du, dv = (u-cx+size/2)%size-size/2, (v-cy+size/2)%size-size/2
            power += _gaussian_plane([200,0,0,.8,1.1,.4,0,0,0], du, dv)
        fit = fit_splodge_centres(power, seeds)
        onp.testing.assert_allclose(fit.centres, truth, atol=.04)
        self.assertTrue(fit.success.all())
        self.assertFalse(fit.at_bound.any())
        self.assertFalse(fit.masks[0].all())
        figs = plot_splodge_fits(fit)
        self.assertEqual(figs[1][1].shape, (6,3))
        import matplotlib.pyplot as plt
        for fig, _ in figs:
            plt.close(fig)

    def test_periodic_patch_at_fourier_boundary(self):
        size = 32
        seeds = onp.array([[15.8, 6.], [-15.8, -7.], [5,10], [-5,10], [8,-9], [-8,-9]])
        truth = seeds + [.35, .22]
        v,u = onp.indices((size,size))
        power = onp.full((size,size),3.)
        for cx,cy in truth:
            du,dv=(u-cx+size/2)%size-size/2,(v-cy+size/2)%size-size/2
            power += 100*onp.exp(-.5*((du/.7)**2+(dv/.9)**2))
        fit = fit_splodge_centres(power,seeds)
        onp.testing.assert_allclose(fit.centres,truth,atol=.01)

    def test_displacement_bound_is_reported(self):
        v,u=onp.indices((32,32))
        power=1+100*onp.exp(-((u-10.)**2+(v-8.)**2)/2)
        fit=fit_splodge_centres(power,onp.tile([6.,8.],(6,1)))
        self.assertTrue(fit.at_bound.all())
        self.assertFalse(onp.allclose(fit.centres,fit.initial_centres))


class PhasorTests(unittest.TestCase):
    def test_known_fractional_phases_and_signed_conjugates(self):
        n=64
        y,x=onp.indices((n,n))-n//2
        centres=onp.tile([8.35,11.4],(6,1))
        envelope=onp.exp(-(x*x+y*y)/(2*8.**2))
        phases=onp.array([.7,-1.1,2.2])
        frames=10*envelope*onp.cos(2*onp.pi*(8.35*x+11.4*y)/n+phases[:,None,None])
        result=extract_phasors(frames,centres)
        onp.testing.assert_allclose(onp.angle(result.phasors[:,0]),phases,atol=1e-4)
        negative=extract_phasors(frames,-centres)
        onp.testing.assert_allclose(negative.phasors,onp.conj(result.phasors),atol=1e-8)

    def test_direct_sum_boundary_and_integer_fft_origin(self):
        frames=onp.random.default_rng(1).normal(size=(3,32,32))
        centres=onp.array([[15.9,6.1],[-16.1,6.1],[-.2,-9.3],[31.8,-9.3],[4,5],[-4,-5]])
        r=extract_phasors(frames,centres)
        onp.testing.assert_allclose(r.phasors[:,0],r.phasors[:,1],atol=1e-10)
        onp.testing.assert_allclose(r.phasors[:,2],r.phasors[:,3],atol=1e-10)
        ft=onp.fft.fft2(onp.fft.ifftshift(frames*_window((32,32),None),axes=(-2,-1)))
        onp.testing.assert_allclose(r.phasors[:,4],ft[:,5,4],atol=1e-10)
        onp.testing.assert_allclose(r.phasors[:,5],onp.conj(ft[:,5,4]),atol=1e-10)

    def test_noise_and_complex_amplitude_snr_scaling(self):
        rng=onp.random.default_rng(22)
        noise=rng.normal(size=(3000,32,32))
        result=extract_phasors(noise,CENTRES)
        expected=onp.sum(_window((32,32),None)**2)
        self.assertAlmostEqual(result.noise_power.mean()/expected,1.,delta=.04)
        onp.testing.assert_allclose(onp.mean(abs(result.phasors)**2,axis=0)/expected,1,atol=.06)
        twice=extract_phasors(noise*2,CENTRES,batch_size=173)
        onp.testing.assert_allclose(twice.noise_power,4*result.noise_power,rtol=1e-12)
        a=phasors_to_pd(result.phasors,result.noise_power)
        b=phasors_to_pd(twice.phasors,twice.noise_power)
        onp.testing.assert_allclose(a.pd_snr,b.pd_snr,rtol=1e-12)
        onp.testing.assert_allclose(a.pd_snr,abs(result.phasors)/onp.sqrt(result.noise_power[:,None]))

    def test_nonfinite_frame_and_invalid_noise_preserve_baselines(self):
        frames=fringe_frames(4)
        frames[1,0,0]=onp.nan
        result=extract_phasors(frames,CENTRES)
        self.assertFalse(result.valid_frames[1])
        self.assertTrue(onp.isnan(result.phasors[1]).all())
        z=onp.full((3,6),20+1j)
        pd=phasors_to_pd(z,[0,onp.nan,1])
        self.assertTrue(onp.isnan(pd.pd_tel[:2]).all())
        self.assertTrue(onp.isfinite(pd.pd_baseline).all())
        onp.testing.assert_array_equal(pd.valid,[False,False,True])


class ProjectionTests(unittest.TestCase):
    def test_wrap_consistency_signs_and_zero_common_piston(self):
        tel=onp.array([[.35,-.35,.1,-.1], [.05,-.04,.02,-.03]])
        baseline=tel@BASELINE_MATRIX.T
        pd=phasors_to_pd(20*onp.exp(2j*onp.pi*baseline),onp.ones(2))
        self.assertTrue(pd.valid.all())
        onp.testing.assert_allclose(pd.pd_tel.sum(axis=1),0,atol=1e-15)
        # Wrapped solutions may differ by telescope integer pistons.
        delta=pd.pd_tel@BASELINE_MATRIX.T-baseline
        onp.testing.assert_allclose(delta-onp.rint(delta),0,atol=1e-14)
        onp.testing.assert_allclose(pd.pd_tel[1],tel[1],atol=1e-14)

    def test_weighted_projection_and_rtc_search_reference(self):
        x=onp.array([.44,-.22,.1,.33,-.4,.21])
        snr=onp.array([8,10,13,18,20,9])
        pd=phasors_to_pd(snr[None]*onp.exp(2j*onp.pi*x),[1])
        w=onp.diag(snr**2)
        proj=onp.linalg.pinv(BASELINE_MATRIX.T@w@BASELINE_MATRIX)@BASELINE_MATRIX.T@w
        chi=[]
        candidates=[]
        for bits in range(64):
            if bits.bit_count()>2:
                continue
            trial=x+onp.where(x>0,-1,1)*((bits>>onp.arange(6))&1)
            candidate=proj@trial
            candidates.append(candidate)
            chi.append(onp.sum(snr**2*(trial-BASELINE_MATRIX@candidate)**2))
        actual=pd.pd_tel[0]
        residual=x-BASELINE_MATRIX@actual
        # Integer telescope pistons give tied wrap-search minima.
        expected=candidates[onp.argmin(chi)]
        delta=BASELINE_MATRIX@(actual-expected)
        onp.testing.assert_allclose(delta-onp.rint(delta),0,atol=1e-13)
        trial=x+onp.where(x>0,-1,1)*onp.rint(onp.abs(residual))
        self.assertAlmostEqual(onp.sum(snr**2*(trial-BASELINE_MATRIX@actual)**2),min(chi),places=10)

    def test_disconnected_and_connected_spanning_tree(self):
        snr=onp.array([[20,0,0,0,0,20],[20,20,20,0,0,0],[4.5]*6])
        pd=phasors_to_pd(snr.astype(complex),onp.ones(3))
        onp.testing.assert_array_equal(pd.connected,[False,True,False])
        self.assertTrue(onp.isnan(pd.pd_tel[[0,2]]).all())
        self.assertTrue(onp.isfinite(pd.pd_baseline).all())


class SpectrumTests(unittest.TestCase):
    def result(self):
        n=80
        fps=100.
        # Two cubes with a large save-time gap and distinct DC levels.
        indices=onp.tile(onp.arange(n),2)
        files=onp.repeat([0,1],n)
        seconds=indices/fps + files*10
        times=onp.datetime64('2026-04-03','ns')+onp.rint(seconds*1e9).astype('timedelta64[ns]')
        values=onp.sin(2*onp.pi*12.5*indices/fps)[:,None]*onp.array([[1,-1,2,-2]])
        values[80:] += onp.array([100,-100,200,-200])
        phase=phasors_to_pd(onp.ones((2*n,6),complex)*20,onp.ones(2*n))
        phase.pd_tel=values
        return ExtractionResult(**vars(phase),times=times,source_files=(),
                                file_indices=files,frame_indices=indices,
                                fps=onp.full(2*n,fps),phasors=onp.empty((2*n,6)),
                                noise_power=onp.ones(2*n),valid_frames=onp.ones(2*n,bool),fit=None)

    def test_within_cube_run_weighting_and_native_rate(self):
        from scipy.signal import welch
        r=self.result()
        r.valid[25]=False
        spectrum=telescope_pd_welch(r,nperseg=16)
        total=0
        weights=0
        for a,b in ((0,25),(26,80),(80,160)):
            _,p=welch(r.pd_tel[a:b],fs=100,nperseg=16,noverlap=8,axis=0)
            k=1+(b-a-16)//8
            total += k*p
            weights += k
        onp.testing.assert_allclose(spectrum.power,total/weights,atol=1e-12)
        self.assertEqual(spectrum.run_count,3)
        self.assertEqual(spectrum.segment_count,weights)
        self.assertEqual(spectrum.fps,100)
        self.assertEqual(spectrum.frequencies[onp.argmax(spectrum.power[:,0])],12.5)

    def test_frame_gaps_time_jitter_and_mixed_rates(self):
        r=self.result()
        r.frame_indices[40:80] += 1
        r.times[120:] += onp.timedelta64(1000,'ns')
        spectrum=telescope_pd_welch(r,nperseg=16)
        self.assertEqual(spectrum.run_count,4)
        with self.assertRaisesRegex(ValueError,'No valid uniform'):
            telescope_pd_welch(r,nperseg=100)
        r.fps[80:] = 101
        with self.assertRaisesRegex(ValueError,'one native FPS'):
            telescope_pd_welch(r,nperseg=16)


class FitsTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)/'20260403'
        self.root.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def write(self,name,frames,compressed=True,mode='GCDS',fps=10.,offset=1000):
        header=fits.Header({'FPS':fps,'UTDATE':'2026-04-03','RO_MODE':mode,'OFFSET':offset})
        path=self.root/name
        data=onp.rint(frames+offset).astype(onp.int32)
        if compressed:
            fits.HDUList([fits.PrimaryHDU(),fits.CompImageHDU(data,header=header,
                          quantize_level=0)]).writeto(path)
        else:
            fits.PrimaryHDU(data,header=header).writeto(path)
        return path

    def test_completed_bounds_include_following_filename_and_timezones(self):
        p=self.write('hei_k1_T01_00_02.000.fits',fringe_frames(20))
        cubes=index_fits([p],timestamp='completed',start='2026-04-03T03:00:00.250+02:00',
                         end='2026-04-03T01:00:01.000Z')
        self.assertEqual((cubes[0].first,cubes[0].stop),(3,10))
        self.assertEqual(cubes[0].times()[0],onp.datetime64('2026-04-03T01:00:00.300'))
        batches=list(read_cube_batches(cubes[0],3))
        self.assertEqual([len(a) for a,_ in batches],[3,3,1])
        exact=index_fits([p],timestamp='completed',start='2026-04-03T01:00:00.300',
                         end='2026-04-03T01:00:00.600')[0]
        self.assertEqual((exact.first,exact.stop),(3,6))
        self.assertEqual(utc64('2026-04-03T03:00:00.300000001+02:00'),
                         onp.datetime64('2026-04-03T01:00:00.300000001','ns'))
        subframe=index_fits([p],timestamp='completed',start='2026-04-03T01:00:00.300000001Z')[0]
        self.assertEqual(subframe.first,4)

    def test_completed_cube_crossing_midnight_and_missing_fps(self):
        p=self.write('hei_k1_T00_00_01.000.fits',fringe_frames(20))
        cube=index_fits([p],timestamp='completed',start='2026-04-02T23:59:59.5Z',
                        end='2026-04-03T00:00:00.5Z')[0]
        self.assertEqual((cube.first,cube.stop),(5,15))
        self.assertEqual(cube.times()[0],onp.datetime64('2026-04-02T23:59:59.5'))
        with fits.open(p,mode='update') as hdus:
            del hdus[1].header['FPS']
        with self.assertRaisesRegex(ValueError,'FPS'):
            index_fits([p],timestamp='completed')

    def test_primary_compressed_extraction_and_batch_agreement(self):
        frames=fringe_frames(60)
        a=self.write('hei_k1_T01_00_06.000.fits',frames)
        b=self.write('hei_k1_T01_00_12.000.fits',frames,False,offset=3000)
        r=extract_pd_from_fits([a,b],CENTRES,batch_size=13,snr_threshold=0)
        other=extract_pd_from_fits([a,b],CENTRES,batch_size=31,snr_threshold=0)
        self.assertEqual(r.phasors.shape,(120,6))
        self.assertEqual(r.fit.frame_count,120)
        onp.testing.assert_array_equal(r.fit.mean_power,other.fit.mean_power)
        onp.testing.assert_allclose(r.phasors,other.phasors,atol=1e-9)
        onp.testing.assert_allclose(r.pd_tel,other.pd_tel,atol=1e-12)
        onp.testing.assert_allclose(r.phasors[:60],r.phasors[60:],atol=1e-9)
        onp.testing.assert_array_equal(r.frame_indices,onp.tile(onp.arange(60),2))
        onp.testing.assert_array_equal(r.file_indices,onp.repeat([0,1],60))
        self.assertTrue(r.valid.all())
        selected=extract_pd_from_fits([a,b],CENTRES,start='2026-04-03T01:00:04',
                                      end='2026-04-03T01:00:08',snr_threshold=0)
        self.assertEqual(len(selected.times),40)

    def test_loader_preserves_start_timestamp_default_for_both_layouts(self):
        frames=fringe_frames(10)
        self.write('hei_k1_T01_00_00.000.fits',frames,False)
        self.write('hei_k1_T01_00_01.000.fits',frames,True)
        from datetime import datetime
        loader=DataBundler(self.root)
        args=('hei_k1',datetime(2026,4,3,1),datetime(2026,4,3,1,0,2))
        _,times,loaded=loader.load_between(*args)
        _,other_times,reduced=loader.apply_fn_to_frames(*args,lambda f:f.mean())
        self.assertEqual(times[0],onp.datetime64('2026-04-03T01:00:00'))
        self.assertEqual(len(loaded),20)
        onp.testing.assert_array_equal(times,other_times)
        onp.testing.assert_allclose(reduced,loaded.mean(axis=(1,2)))

    def test_ndmr_rejected_and_failed_fit_diagnostics_retained(self):
        p=self.write('hei_k1_T01_00_01.000.fits',fringe_frames(10),mode='NDMR')
        with self.assertRaisesRegex(ValueError,'NDMR reset-validity'):
            extract_pd_from_fits([p],CENTRES)
        p=self.write('hei_k1_T01_00_02.000.fits',onp.zeros((10,32,32)))
        with self.assertRaises(SplodgeFitError) as error:
            extract_pd_from_fits([p],onp.zeros((6,2)))
        self.assertEqual(error.exception.fit.mean_power.shape,(32,32))
        import matplotlib.pyplot as plt
        for figure, _ in plot_splodge_fits(error.exception.fit):
            figure.canvas.draw()
            plt.close(figure)
        flat=fit_splodge_centres(onp.zeros((32,32)),CENTRES)
        self.assertFalse(flat.success.any())


if __name__=='__main__':
    unittest.main()
