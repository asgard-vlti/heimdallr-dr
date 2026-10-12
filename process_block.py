"""
As a test, we will manually process a block of data, computing
V^2 and closure phases.
"""
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from astropy.io import fits
from _script_timestamps import parse_time_seconds_and_str as _parse_time_seconds_and_str
plt.ion()

#---- Inputs ----
#tstart = "06:08:45"  # Start time of the data block
#tstop = "06:10:50"   # Stop time of the data block

#tstart = "06:14:28"  # Start time of the data block
#tstop = "06:16:29"   # Stop time of the data block

#tstart = "06:21:39"  # Start time of the data block
#tstop = "06:23:45"   # Stop time of the data block

#!!! This next block doesn't work as there is no dark data!!!
#tstart = "06:27:29"  # Start time of the data block
#tstop = "06:30:30"   # Stop time of the data block

#tstart = "06:36:13"  # Start time of the data block
#tstop = "06:38:15"   # Stop time of the data block

#tstart = "06:57:59"  # Start time of the data block
#tstop = "06:59:50"   # Stop time of the data block

dir = "/Users/mireland/data/heimdallr/HD222805/"  # Directory containing data files

dir = "/Users/mireland/data/heimdallr/HD5877/"  # Directory containing data files

# List of target, start and stop times to process
obs = [["HD5877", "06:10:27", "06:12:29"],
       ["HD5877", "06:20:21", "06:22:22"],
       ["HD5877", "06:28:51", "06:30:57"],
       ["HD5877", "06:33:56", "06:36:07"],
       ["HD5877", "06:40:36", "06:42:48"],
       ["HD5877", "06:48:45", "06:49:59"],
       ["HD5877", "06:54:07", "06:56:09"],
       ["HD5877", "06:58:49", "07:00:50"]]

""" tstart = "06:10:27"
tstop = "06:12:29"

tstart = "06:20:21"
tstop = "06:22:22"

tstart = "06:28:51"
tstop = "06:30:57"

tstart = "06:33:56"
tstop = "06:36:07"

tstart = "06:40:36"
tstop = "06:42:48"

tstart = "06:48:45"
tstop = "06:49:59"

tstart = "06:54:07"
tstop = "06:56:09"

tstart = "06:58:49"
tstop = "07:00:50"
 """
smooth = 25  # Smoothing window size
gd_smooth = 25  # Smoothing window size for group delay
gap = 250  # Gap size for dropout detection
good_threshold = 1.0  # Threshold multiplier for good data selection

#---- Parameters that should come from the toml file ----
hole_diameter = 9e-3 # Hole diameter in radians, converging to the detector
beam_x = [-0.00792,-0.0085,  0.0 ,  0.02130]  # X positions of beams on the detector (m)
beam_y = [0.019,  -0.01484,  0.0 , -0.00015]  # Y positions of beams on the detector (m)
# Revised based on the power spectrum. By hand...
beam_x = [-0.0095,-0.010,  0.0 ,  0.018]  # X positions of beams on the detector (m)
beam_y = [0.0175,  -0.014,  0.0 , -0.001]  # Y positions of beams on the detector (m)

beam_x = [-0.008,-0.009,  0.0 ,  0.0195]  # X positions of beams on the detector (m)
beam_y = [0.0175,  -0.015,  0.0 , -0.000]  # Y positions of beams on the detector (m)

pix = 24
K1wave = 2.07
K2wave = 2.21

#--- Constant Matrices ---
N_TEL = 4
N_BL  = N_TEL*(N_TEL-1)//2 # i.e. 6
bl2beam = np.array([[0,1],
                    [0,2],
                    [0,3],
                    [1,2],
                    [1,3],
                    [2,3]]).astype(int)

# Remove top-level single-block processing. Wrap it into a function instead.
def process_block(tstart: str, tstop: str, dir_path: str, target: str | None = None):
    # Parse the requested time range for this block
    tstart_sec = _parse_time_seconds_and_str("T" + tstart)[0]
    tstop_sec = _parse_time_seconds_and_str("T" + tstop)[0]

    # Determine files for this block
    k1_files = sorted(
        glob(dir_path + "hei_k1*.fits"),
        key=lambda path: (_parse_time_seconds_and_str(path) or (float("inf"), ""))[0],
    )
    k2_files = sorted(glob(dir_path + "hei_k2*.fits"))

    # --- Begin existing block processing (was top-level code) ---
    print("Reading data files...")
    block_fn_k1 = []
    block_fn_k2 = []
    fluxes_k1 = []
    fluxes_k2 = []
    for fn in k1_files:
        parsed = _parse_time_seconds_and_str(fn)
        if parsed is None:
            continue
        t_sec, t_str = parsed
        if tstart_sec <= t_sec <= tstop_sec:
            print(f"Initial Processing file: {fn} at time {t_str}")
            dd = fits.getdata(fn)
            # Median over each spatial dimension
            flux_k1 = np.median(dd, axis=(1,2))
            fluxes_k1.append(flux_k1)
            fn_k2 = fn.replace("_k1_", "_k2_")
            try:
                dd2 = fits.getdata(fn_k2)
            except FileNotFoundError:
                print(f"File not found: {fn_k2}")
                continue
            flux_k2 = np.median(dd2, axis=(1,2))
            fluxes_k2.append(flux_k2)
            block_fn_k1.append(fn)
            block_fn_k2.append(fn_k2)
    fluxes_k1 = np.array(fluxes_k1)
    fluxes_k2 = np.array(fluxes_k2)

    # Let's use the smoothed sum of fluxes
    fluxes_sum = fluxes_k1.flatten() + fluxes_k2.flatten() - 2000
    sflux = np.convolve(fluxes_sum, np.ones(smooth)/smooth, mode='same')
    sflux[:12] = sflux[12]
    sflux[-12:] = sflux[-13]

    # Plot the smoothed flux. This will be over-written later, but useful for debugging.
    plt.figure(1)
    plt.clf()
    plt.plot(sflux, label='Smoothed Flux')
    plt.draw()

    #Median flux. This is towards the lower part of the on-fringe range
    medflux = np.median(sflux) 
    dk = np.where(sflux < 0.3*0.25*medflux)[0]
    if len(dk) == 0:
        raise ValueError("No dark frames found; cannot proceed with processing.")
    first_dk = np.where(((dk[1:]-dk[:-1]) != 1))[0][-1] + 1
    dk = dk[gap + first_dk:]

    bright = np.where(sflux > 0.6*medflux)[0]
    bright = bright[:-gap]
    # Check that we don't have dropouts.
    num_dropouts = np.sum(np.diff(bright) > 1)
    if num_dropouts > 0:
        raise ValueError("Data has dropouts; cannot proceed with processing.")
    kappa = np.arange(len(bright) + gap,dk[0]-gap)
    kappa_ixs = []
    #last_ix is the last index of a downwards transition.
    last_ix = 0 
    for i in range(4):
        ix = int( (0.125 + i*0.25) * len(kappa) )
        # Find the start of this kappa region
        w = np.where(sflux[kappa[last_ix+gap:]] > 0.5* sflux[kappa[ix]])[0]
        this_kappa_end = np.where((w[1:]-w[:-1]) != 1)[0]
        if len(this_kappa_end) == 0:
            this_kappa_end = len(w)
            this_kappa_start = 0
        else:
            this_kappa_end = this_kappa_end[0]
            this_kappa_start = w[0]
        next_ix = last_ix + 2*gap + this_kappa_end
        first_good = last_ix + 2*gap + this_kappa_start
        kappa_ixs.append(kappa[np.arange(first_good,next_ix-2*gap)])
        last_ix = next_ix
        
    plt.figure(1)
    plt.clf()
    plt.plot(sflux, label='Smoothed Flux')
    plt.plot(bright, sflux[bright], 'g.', label='Bright Frames')
    plt.plot(dk, sflux[dk], 'b.', label='Dark Frames')
    for i, kix in enumerate(kappa_ixs):
        plt.plot(kix, sflux[kix], f'C{i+1}.', label=f'Kappa Beam {i+1}')
    plt.xlabel('Frame Index')
    plt.ylabel('Flux')
    plt.legend()
    plt.title('Flux Analysis with Bright, Dark, and Kappa Frames')

    nfiles = fluxes_k1.shape[0]
    nfr = fluxes_k1.shape[1]
    szy = dd.shape[1]
    szx = dd.shape[2]
    dark1 = np.zeros((szy, szx))
    dark2 = np.zeros((szy, szx))

    #Process the dark regions
    current_f_ix = -1
    for ix in dk:
        f_ix = ix // nfr
        if f_ix != current_f_ix:
            d1 = fits.getdata(block_fn_k1[f_ix])
            d2 = fits.getdata(block_fn_k2[f_ix])
            current_f_ix = f_ix
        fr = ix % nfr
        dark1 += d1[fr]
        dark2 += d2[fr]
    dark1 /= len(dk)
    dark2 /= len(dk)

    #Process the kappa regions to get 4 average images.
    kappa_imgs1 = np.zeros((4, szy, szx))
    kappa_imgs2 = np.zeros((4, szy, szx))
    current_f_ix = -1
    for i, kix in enumerate(kappa_ixs):
        for ix in kix:
            f_ix = ix // nfr
            if f_ix != current_f_ix:
                d1 = fits.getdata(block_fn_k1[f_ix])
                d2 = fits.getdata(block_fn_k2[f_ix])
                current_f_ix = f_ix
            fr = ix % nfr
            kappa_imgs1[i] += d1[fr] - dark1
            kappa_imgs2[i] += d2[fr] - dark2
        kappa_imgs1[i] /= len(kix)
        kappa_imgs2[i] /= len(kix)

    # Diplay the kappa images in a 4x2 grid
    plt.figure(2)
    plt.clf()
    for i in range(4):
        plt.subplot(2,4,i+1)
        plt.imshow(kappa_imgs1[i], origin='lower', cmap='inferno')
        plt.title(f'Kappa Beam {i+1} K1')
        plt.colorbar()
        plt.subplot(2,4,i+5)
        plt.imshow(kappa_imgs2[i], origin='lower', cmap='inferno')
        plt.title(f'Kappa Beam {i+1} K2')
        plt.colorbar()
    plt.suptitle('Kappa Average Images')
    plt.tight_layout()

    # Create a window function
    xy = np.meshgrid(np.arange(szx), np.arange(szy))
    rr = np.sqrt((xy[0]-szx//2)**2  + (xy[1] - szy//2)**2)
    window = np.exp(-(rr/(szx//2))**8)

    # Find the V^2 scaling from the kappa images, i.e. based
    # on overlap.
    V2s_1 = np.zeros(N_BL)
    V2s_2 = np.zeros(N_BL)
    kappa_win1 = kappa_imgs1 * window
    kappa_win2 = kappa_imgs2 * window
    kappa_win1 /= np.sqrt(np.sum(np.sum(kappa_win1, axis=1)**2))
    kappa_win2 /= np.sqrt(np.sum(np.sum(kappa_win2, axis=1)**2))
    for i in range(N_BL):
        V2s_1[i] = np.sum(kappa_win1[bl2beam[i,0]]*kappa_win1[bl2beam[i,1]])
        V2s_2[i] = np.sum(kappa_win2[bl2beam[i,0]]*kappa_win2[bl2beam[i,1]])

    # Make a power spectrum from typical frames of bright data
    N1 = 1000
    N2 = 1100
    ps1 = np.zeros((szy, szx))
    ps2 = np.zeros((szy, szx))
    current_f_ix = -1
    for ix in bright[N1:N2]:
        f_ix = ix // nfr
        if f_ix != current_f_ix:
            d1 = fits.getdata(block_fn_k1[f_ix])
            d2 = fits.getdata(block_fn_k2[f_ix])
            current_f_ix = f_ix
        fr = ix % nfr
        frame1 = (d1[fr] - dark1) * window
        frame2 = (d2[fr] - dark2) * window
        ft1 = np.fft.fft2(frame1)
        ft2 = np.fft.fft2(frame2)
        ps1 += np.abs(ft1)**2
        ps2 += np.abs(ft2)**2
    ps1 /= (N2 - N1)
    ps2 /= (N2 - N1)
    ps1 = np.fft.fftshift(ps1)
    ps2 = np.fft.fftshift(ps2)

    #--- Simple geometry ---
    #bl_x is the difference in beam_x for the baselines
    bl_x = np.take(beam_x, bl2beam[:, 1]) - np.take(beam_x, bl2beam[:, 0])
    bl_y = np.take(beam_y, bl2beam[:, 1]) - np.take(beam_y, bl2beam[:, 0])

    #Convert to pixels
    bl_u1 = bl_x * pix / K1wave * szx 
    bl_v1 = bl_y * pix / K1wave * szy
    bl_u2 = bl_x * pix / K2wave * szx 
    bl_v2 = bl_y * pix / K2wave * szy

    # Also create interpolation positions
    p0_K1 = np.zeros(N_BL, dtype=int)
    p1_K1 = np.zeros(N_BL, dtype=int)
    p0_K2 = np.zeros(N_BL, dtype=int)
    p1_K2 = np.zeros(N_BL, dtype=int)
    q0_K1 = np.zeros(N_BL, dtype=int)
    q1_K1 = np.zeros(N_BL, dtype=int)
    q0_K2 = np.zeros(N_BL, dtype=int)
    q1_K2 = np.zeros(N_BL, dtype=int)
    dp_K1 = np.zeros(N_BL)
    dq_K1 = np.zeros(N_BL)
    dp_K2 = np.zeros(N_BL)
    dq_K2 = np.zeros(N_BL)
    for i in range(N_BL):
        p0_K1[i] = int(szx//2 + bl_u1[i])
        q0_K1[i] = int(szy//2 + bl_v1[i])
        p1_K1[i] = p0_K1[i] + 1
        q1_K1[i] = q0_K1[i] + 1
        dp_K1[i] = (szx//2 + bl_u1[i]) - p0_K1[i]
        dq_K1[i] = (szy//2 + bl_v1[i]) - q0_K1[i]
        p0_K2[i] = int(szx//2 + bl_u2[i])
        q0_K2[i] = int(szy//2 + bl_v2[i])
        p1_K2[i] = p0_K2[i] + 1
        q1_K2[i] = q0_K2[i] + 1
        dp_K2[i] = (szx//2 + bl_u2[i]) - p0_K2[i]
        dq_K2[i] = (szy//2 + bl_v2[i]) - q0_K2[i]

    #--- Plot the power spectra with baseline positions ---
    plt.figure(3)
    plt.clf()
    plt.subplot(1,2,1)
    plt.imshow(np.log10(ps1/ps1[szx//2, szy//2]+0.001), origin='lower', cmap='inferno')
    plt.colorbar()
    plt.title('Power Spectrum K1')
    for i in range(len(bl_u1)):
        plt.plot(szx//2 + bl_u1[i], szy//2 + bl_v1[i], 'go')
        plt.plot(szx//2 - bl_u1[i], szy//2 - bl_v1[i], 'go')
    plt.subplot(1,2,2)
    plt.imshow(np.log10(ps2/ps2[szx//2, szy//2]+0.001), origin='lower', cmap='inferno')
    plt.colorbar()
    plt.title('Power Spectrum K2')
    for i in range(len(bl_u2)):
        plt.plot(szx//2 + bl_u2[i], szy//2 + bl_v2[i], 'go')
        plt.plot(szx//2 - bl_u2[i], szy//2 - bl_v2[i], 'go')
    plt.suptitle('Power Spectra with Baseline Positions')   

    #Now extract complex visibilities from all bright frames!
    V1 = np.zeros((len(bright), N_BL), dtype=complex)
    V2 = np.zeros((len(bright), N_BL), dtype=complex)
    current_f_ix = -1
    for count, ix in enumerate(bright):
        f_ix = ix // nfr
        if f_ix != current_f_ix:
            d1 = fits.getdata(block_fn_k1[f_ix])
            d2 = fits.getdata(block_fn_k2[f_ix])
            current_f_ix = f_ix
        fr = ix % nfr
        frame1 = (d1[fr] - dark1) * window
        frame2 = (d2[fr] - dark2) * window
        ft1 = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(frame1)))
        ft2 = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(frame2)))
        for i in range(N_BL):
            # Extract the visibility at the baseline location, 
            # interpolating linearly between nearest 4 pixels.
            V1[count, i] = ft1[q0_K1[i], p0_K1[i]]*(1 - dp_K1[i])*(1 - dq_K1[i]) + \
                         ft1[q0_K1[i], p1_K1[i]]*(dp_K1[i])*(1 - dq_K1[i]) + \
                         ft1[q1_K1[i], p0_K1[i]]*(1 - dp_K1[i])*(dq_K1[i]) + \
                         ft1[q1_K1[i], p1_K1[i]]*(dp_K1[i])*(dq_K1[i])
            V1[count, i] /= ft1[szx//2, szy//2]
            V2[count, i] = ft2[q0_K2[i], p0_K2[i]]*(1 - dp_K2[i])*(1 - dq_K2[i]) + \
                         ft2[q0_K2[i], p1_K2[i]]*(dp_K2[i])*(1 - dq_K2[i]) + \
                         ft2[q1_K2[i], p0_K2[i]]*(1 - dp_K2[i])*(dq_K2[i]) + \
                         ft2[q1_K2[i], p1_K2[i]]*(dp_K2[i])*(dq_K2[i])
            V2[count, i] /= ft2[szx//2, szy//2]

    # Compute V^2. 
    V2_1 = np.abs(V1)**2
    V2_2 = np.abs(V2)**2

    # Compute group delay phasors
    GD_phasor = V1*np.conj(V2)
    # Smooth the group delay phasors
    GD_phasor_smooth = np.zeros_like(GD_phasor)
    for i in range(N_BL):
        re_part = np.convolve(GD_phasor[:, i].real, np.ones(gd_smooth)/gd_smooth, mode='same')
        im_part = np.convolve(GD_phasor[:, i].imag, np.ones(gd_smooth)/gd_smooth, mode='same')
        GD_phasor_smooth[:, i] = re_part + 1j*im_part
        
    # Find the foreground from the maximum over all baselines.
    high_GD = np.max(np.abs(GD_phasor_smooth), axis=1) 
    no_fringes = np.where(high_GD < 0.1 * np.mean(high_GD))[0]
    no_fringes_gap = np.where(np.diff(no_fringes) != 1)[0]
    if len(no_fringes_gap) > 0:
        foreground = no_fringes[no_fringes_gap[-1]+1:]
    else:
        foreground = no_fringes
    if len(foreground) < 1000:
        raise ValueError("Not enough foreground data to compute reliable V^2.")
    foreground = foreground[gd_smooth:]

    # Find the good frames based on the 3rd lowest GD_phasor value
    gd_mags = np.abs(GD_phasor_smooth)
    third_lowest = np.sort(gd_mags, axis=1)[:,2]
    good_data = np.where(third_lowest > good_threshold*np.mean(third_lowest[:foreground[0]]))[0]

    # Plot the GD phasor third_lowest and high_GD for inspection,
    # marking the foreground and good data regions.
    plt.figure(4)
    plt.clf()
    plt.plot(high_GD, label='Max GD Phasor Magnitude')
    plt.plot(third_lowest, label='3rd Lowest GD Phasor Magnitude')
    plt.plot(foreground, high_GD[foreground], 'r.', label='Foreground Frames')
    plt.plot(good_data, third_lowest[good_data], 'go', label='Good Data Frames')
    plt.xlabel('Frame Index')
    plt.ylabel('GD Phasor Magnitude')
    plt.legend()
    plt.title('Group Delay Phasor Analysis')

    # Bias-correct V^2 and correct for flux imbalance
    V2_1 = V2_1 - np.mean(V2_1[foreground,:], axis=0)
    V2_2 = V2_2 - np.mean(V2_2[foreground,:], axis=0)
    V2_1 /= V2s_1
    V2_2 /= V2s_2

    # Average V^2 over good data
    mean_V2_1 = np.mean(V2_1[good_data,:], axis=0)
    mean_V2_2 = np.mean(V2_2[good_data,:], axis=0)

    # Print out the final V^2 results
    print(f"Results for {target or ''} {tstart}-{tstop}".strip())
    print("Baseline\tV^2 K1\tV^2 K2")
    for i in range(N_BL):
        print(f"{bl2beam[i,0]+1}-{bl2beam[i,1]+1}\t{mean_V2_1[i]:.4f}\t{mean_V2_2[i]:.4f}")
    print("CSV Format:")
    csv_line = ",".join([f"{mean_V2_1[i]:.4f}" for i in range(N_BL)]) + "," + \
               ",".join([f"{mean_V2_2[i]:.4f}" for i in range(N_BL)])
    print(csv_line)

    # Now compute closure phases
    cp2bl = np.array([[0, 3, 1],
                      [0, 4, 2],
                      [1, 5, 2],
                      [3, 5, 4]])
    N_CP = cp2bl.shape[0]

    #Bispectra for closure phase
    bs1 = np.zeros((len(good_data), N_CP), dtype=complex)
    bs2 = np.zeros((len(good_data), N_CP), dtype=complex)

    for count, ix in enumerate(good_data):
        for i in range(N_CP):
            b1 = cp2bl[i,0]
            b2 = cp2bl[i,1]
            b3 = cp2bl[i,2]
            bs1[count, i] = V1[ix, b1] * V1[ix, b2] * np.conj(V1[ix, b3])
            bs2[count, i] = V2[ix, b1] * V2[ix, b2] * np.conj(V2[ix, b3])
    mean_bs1 = np.mean(bs1, axis=0)
    mean_bs2 = np.mean(bs2, axis=0)
    cp_1 = np.angle(mean_bs1, deg=True)
    cp_2 = np.angle(mean_bs2, deg=True)
    # Print out closure phases, also in csv format
    print("Closure Phase Results (degrees):")
    print("Triangle\tCP K1\tCP K2")
    for i in range(N_CP):
        b1 = cp2bl[i,0]
        b2 = cp2bl[i,1]
        b3 = cp2bl[i,2]
        print(f"{b1+1}-{b2+1}-{b3+1}\t{cp_1[i]:.2f}\t{cp_2[i]:.2f}")
    print("CSV Format:")
    csv_line_cp = ",".join([f"{cp_1[i]:.2f}" for i in range(N_CP)]) + "," + \
                  ",".join([f"{cp_2[i]:.2f}" for i in range(N_CP)])
    print(csv_line_cp)

    # Return a compact result for this block
    return {
        "V2_k1": mean_V2_1,
        "V2_k2": mean_V2_2,
        "cp_k1": cp_1,
        "cp_k2": cp_2,
    }

# --- Loop through all observations ---
results = []
for target, tstart, tstop in obs:
    print(f"Processing block: {target} {tstart}-{tstop}")
    try:
        out = process_block(tstart, tstop, dir, target=target)
        results.append({
            "target": target,
            "tstart": tstart,
            "tstop": tstop,
            **out
        })
    except Exception as e:
        print(f"Failed block {target} {tstart}-{tstop}: {e}")
# Optionally, results now contains per-block outputs
