import pytest
import numpy as np
import jax
import jax.numpy as npj
from ipie.addons.eph.trial_wavefunction.variational.dd1_old import dD1Variational
from ipie.systems import Generic
from ipie.addons.eph.hamiltonians.holstein import HolsteinModel
from ipie.addons.eph.hamiltonians.ssh import OpticalSSHModel
from ipie.addons.eph.trial_wavefunction.variational.jax.toyozawa import overlap_degeneracy
np.random.seed(125)

def boson(x: np.ndarray, dd1_obj, iperm):
    shift, c0a, c0b = dd1_obj.unpack_x(x)
    shift = np.squeeze(shift)
    c0a = np.squeeze(c0a)
    shift_i = npj.roll(shift, shift=(-iperm, -iperm), axis=(0, 1))
    fac_i = overlap_degeneracy(dd1_obj.ham, iperm) * dd1_obj.Kcoeffs[0].conj() * dd1_obj.Kcoeffs[iperm]
    Ga_i = npj.outer(c0a.conj(), c0a[dd1_obj.perms[iperm]])

    cs_ovlp = dd1_obj.cs_overlap(shift, shift_i)
    phonon_contrib = dd1_obj.ham.w0 * npj.einsum("ij,j->", shift.conj() * shift_i, cs_ovlp.diagonal() * Ga_i.diagonal())
    return (fac_i * phonon_contrib).real

def boson_tot(x: np.ndarray, dd1_obj):
    bos = 0.
    for iperm in range(dd1_obj.nperms):
        bos += boson(x, dd1_obj, iperm)
    return bos

def analytical_hess_bos(x, dd1_obj, iperm):
    shift, c0a, c0b = dd1_obj.unpack_x(x)
    shift = np.squeeze(shift)
    c0a = np.squeeze(c0a)
    beta_i = np.roll(shift, shift=(-iperm, -iperm), axis=(0, 1))
    psia_i = c0a[dd1_obj.perms[iperm]]

    perm_mat = np.roll(np.eye(dd1_obj.ham.N), shift=-iperm, axis=0)
    cs_ovlp = dd1_obj.cs_overlap(shift, beta_i)
    fac_i = overlap_degeneracy(dd1_obj.ham, iperm) * dd1_obj.Kcoeffs[0].conj() * dd1_obj.Kcoeffs[iperm]
    Ga_i = np.outer(c0a.conj(), psia_i)
    ovlp_contracted = np.einsum("i,ij->ij", cs_ovlp.diagonal(), perm_mat)

    d_shift_perm_r = np.einsum("ei,pj->peij", np.eye(dd1_obj.ham.N), perm_mat.dot(shift).dot(perm_mat.T))
    d_shift_perm_r += np.einsum("pi,je->peij", perm_mat.T.dot(shift.conj()), perm_mat)

    d_cs_ovlp_r = np.einsum("peij,ij->peij", d_shift_perm_r, cs_ovlp)
    d_cs_ovlp_r -= np.einsum("ei,pi,ij->peij", np.eye(dd1_obj.ham.N), shift.real, cs_ovlp)
    d_cs_ovlp_r -= np.einsum("kp,ej,kj,ij->peij", perm_mat, perm_mat.T, beta_i.real, cs_ovlp)

    d2_cs_ovlp_r = np.einsum('peij,peij->peij', d_shift_perm_r, d_cs_ovlp_r)
    d2_cs_ovlp_r -= np.einsum("ei,pi,peij->peij", np.eye(dd1_obj.ham.N), shift.real, d_cs_ovlp_r)
    d2_cs_ovlp_r -= np.einsum("kp,ej,kj,peij->peij", perm_mat, perm_mat.T, beta_i.real, d_cs_ovlp_r) #do these need 1/2?
    d2_cs_ovlp_r -= np.einsum('ei,ij,p->peij', np.eye(dd1_obj.ham.N), cs_ovlp, np.ones(dd1_obj.ham.N))
    d2_cs_ovlp_r -= np.einsum('kp,ej,ij->peij',perm_mat, perm_mat.T, cs_ovlp)
    d2_cs_ovlp_r += 2 * np.einsum('ei,pp,ej,ij->peij', np.eye(dd1_obj.ham.N), perm_mat, perm_mat.T, cs_ovlp)

    d_shift_perm_i = -1j * np.einsum("ei,pj->peij", np.eye(dd1_obj.ham.N), perm_mat.dot(shift).dot(perm_mat.T))
    d_shift_perm_i += 1j * np.einsum("pi,je->peij", perm_mat.T.dot(shift.conj()), perm_mat)

    d_cs_ovlp_i = np.einsum("peij,ij->peij", d_shift_perm_i, cs_ovlp)
    d_cs_ovlp_i -= np.einsum("ei,pi,ij->peij", np.eye(dd1_obj.ham.N), shift.imag, cs_ovlp)
    d_cs_ovlp_i += (1j * 0.5 * np.einsum("pb,ej,jb,ij->peij", shift, perm_mat.T, perm_mat, cs_ovlp, optimize=True))
    d_cs_ovlp_i -= 1j * 0.5 * np.einsum("pb,bj,je,ij->peij", shift.conj(), perm_mat.T, perm_mat, cs_ovlp, optimize=True)

    d2_cs_ovlp_i = np.einsum('peij,peij->peij', d_shift_perm_i, d_cs_ovlp_i)
    d2_cs_ovlp_i -= np.einsum("ei,pi,peij->peij", np.eye(dd1_obj.ham.N), shift.imag, d_cs_ovlp_i)
    d2_cs_ovlp_i += 1j * 0.5 * np.einsum("pb,ej,jb,peij->peij", shift, perm_mat.T, perm_mat, d_cs_ovlp_i, optimize=True)
    d2_cs_ovlp_i -= 1j * 0.5 * np.einsum("pb,bj,je,peij->peij", shift.conj(), perm_mat.T, perm_mat, d_cs_ovlp_i, optimize=True)
    d2_cs_ovlp_i -= np.einsum('ei,ij,p->peij', np.eye(dd1_obj.ham.N), cs_ovlp, np.ones(dd1_obj.ham.N))
    d2_cs_ovlp_i -= np.einsum('kp,ej,ij->peij',perm_mat, perm_mat.T, cs_ovlp)
    d2_cs_ovlp_i += 2 * np.einsum('ei,pp,ej,ij->peij', np.eye(dd1_obj.ham.N), perm_mat, perm_mat.T, cs_ovlp)

    w_contracted = dd1_obj.ham.w0 * np.einsum("ij,j,jn->jn", shift.conj() * beta_i, cs_ovlp.diagonal(), perm_mat)
   
    # Shift Hess Real
    boson_contrib = np.einsum("ii,ki,peii->pe", Ga_i, shift.conj() * beta_i, d2_cs_ovlp_r)
    boson_contrib += 2 * np.einsum("ii,peii,peii->pe", Ga_i, d_shift_perm_r, d_cs_ovlp_r, optimize=True)
    boson_contrib += 2 * np.einsum("ii,ei,pp,ei,ii->pe", Ga_i, np.eye(dd1_obj.ham.N), perm_mat, perm_mat.T, cs_ovlp, optimize=True)
    boson_contrib *= dd1_obj.ham.w0

    boson_tmp = [boson_contrib.real + 0j]

    # Shift Hess Imag
    boson_contrib = np.einsum("ii,ki,peii->pe", Ga_i, shift.conj() * beta_i, d2_cs_ovlp_i)
    boson_contrib += 2 * np.einsum("ii,peii,peii->pe", Ga_i, d_shift_perm_i, d_cs_ovlp_i, optimize=True)
    boson_contrib += 2 * np.einsum("ii,ei,pp,ei,ii->pe", Ga_i, np.eye(dd1_obj.ham.N), perm_mat, perm_mat.T, cs_ovlp, optimize=True)
    boson_contrib *= dd1_obj.ham.w0

    boson_tmp[0] += 1j * boson_contrib.real

    # Elec Hess Real
    boson_contrib = 2 * w_contracted.diagonal() 
    boson_tmp.append(boson_contrib.real + 0j)

    # Elec Hess Imag
    boson_contrib = 2 * w_contracted.diagonal()
    boson_tmp[1] += 1j * boson_contrib.real

    bos_x = dd1_obj.pack_x(boson_tmp[0].ravel(), boson_tmp[1].ravel())
    bos_en_hess = (fac_i * bos_x).real

    # PACKING SEEMS TRICKY, AS OVLP CONTRIBS MAY BE COMPLEX VALUED, SO PARTS WILL BE REDISTRIBUTED IN x.
    return bos_en_hess
   

def analytical_hess_bos_tot(x, dd1_obj):
    bos_en_hess = np.zeros_like(x, dtype=np.float64)
    for iperm in range(dd1_obj.nperms):
        bos_en_hess += analytical_hess_bos(x, dd1_obj, iperm)
    return bos_en_hess
   

def elph(x: np.ndarray, dd1_obj, iperm):
    shift, c0a, c0b = dd1_obj.unpack_x(x)
    shift = np.squeeze(shift)
    c0a = np.squeeze(c0a)
    shift_i = npj.roll(shift, shift=(-iperm, -iperm), axis=(0, 1))
    fac_i = overlap_degeneracy(dd1_obj.ham, iperm) * dd1_obj.Kcoeffs[0].conj() * dd1_obj.Kcoeffs[iperm]
    Ga_i = npj.outer(c0a.conj(), c0a[dd1_obj.perms[iperm]])

    cs_ovlp = dd1_obj.cs_overlap(shift, shift_i)
    el_ph_contrib = npj.einsum("ijk,ij,ki,ij->", dd1_obj.ham.g_tensor, Ga_i, shift.conj(), cs_ovlp)
    el_ph_contrib += npj.einsum("ijk,ij,kj,ij->", dd1_obj.ham.g_tensor, Ga_i, shift_i, cs_ovlp)
    return (fac_i * el_ph_contrib).real

def elph_tot(x: np.ndarray, dd1_obj):
    el_ph = 0.
    for iperm in range(dd1_obj.nperms):
        el_ph += elph(x, dd1_obj, iperm)
    return el_ph

def analytical_hess_elph(x, dd1_obj, iperm):
    # Get elph_ij_contracted, g_contracted
    shift, c0a, c0b = dd1_obj.unpack_x(x)
    shift = np.squeeze(shift)
    c0a = np.squeeze(c0a)
    beta_i = np.roll(shift, shift=(-iperm, -iperm), axis=(0, 1))
    psia_i = c0a[dd1_obj.perms[iperm]]

    perm_mat = np.roll(np.eye(dd1_obj.ham.N), shift=-iperm, axis=0)
    cs_ovlp = dd1_obj.cs_overlap(shift, beta_i)
    fac_i = overlap_degeneracy(dd1_obj.ham, iperm) * dd1_obj.Kcoeffs[0].conj() * dd1_obj.Kcoeffs[iperm]
    Ga_i = np.outer(c0a.conj(), psia_i)
    ovlp_contracted = np.einsum("i,ij->ij", cs_ovlp.diagonal(), perm_mat)

    d_shift_perm_r = np.einsum("ei,pj->peij", np.eye(dd1_obj.ham.N), perm_mat.dot(shift).dot(perm_mat.T))
    d_shift_perm_r += np.einsum("pi,je->peij", perm_mat.T.dot(shift.conj()), perm_mat)

    d_cs_ovlp_r = np.einsum("peij,ij->peij", d_shift_perm_r, cs_ovlp)
    d_cs_ovlp_r -= np.einsum("ei,pi,ij->peij", np.eye(dd1_obj.ham.N), shift.real, cs_ovlp)
    d_cs_ovlp_r -= np.einsum("kp,ej,kj,ij->peij", perm_mat, perm_mat.T, beta_i.real, cs_ovlp)

    d2_cs_ovlp_r = np.einsum('peij,peij->peij', d_shift_perm_r, d_cs_ovlp_r)
    d2_cs_ovlp_r -= np.einsum("ei,pi,peij->peij", np.eye(dd1_obj.ham.N), shift.real, d_cs_ovlp_r)
    d2_cs_ovlp_r -= np.einsum("kp,ej,kj,peij->peij", perm_mat, perm_mat.T, beta_i.real, d_cs_ovlp_r) #do these need 1/2?
    d2_cs_ovlp_r -= np.einsum('ei,ij,p->peij', np.eye(dd1_obj.ham.N), cs_ovlp, np.ones(dd1_obj.ham.N))
    d2_cs_ovlp_r -= np.einsum('kp,ej,ij->peij',perm_mat, perm_mat.T, cs_ovlp)
    d2_cs_ovlp_r += 2 * np.einsum('ei,pp,ej,ij->peij', np.eye(dd1_obj.ham.N), perm_mat, perm_mat.T, cs_ovlp)

    d_shift_perm_i = -1j * np.einsum("ei,pj->peij", np.eye(dd1_obj.ham.N), perm_mat.dot(shift).dot(perm_mat.T))
    d_shift_perm_i += 1j * np.einsum("pi,je->peij", perm_mat.T.dot(shift.conj()), perm_mat)

    d_cs_ovlp_i = np.einsum("peij,ij->peij", d_shift_perm_i, cs_ovlp)
    d_cs_ovlp_i -= np.einsum("ei,pi,ij->peij", np.eye(dd1_obj.ham.N), shift.imag, cs_ovlp)
    d_cs_ovlp_i += (1j * 0.5 * np.einsum("pb,ej,jb,ij->peij", shift, perm_mat.T, perm_mat, cs_ovlp, optimize=True))
    d_cs_ovlp_i -= 1j * 0.5 * np.einsum("pb,bj,je,ij->peij", shift.conj(), perm_mat.T, perm_mat, cs_ovlp, optimize=True)

    d2_cs_ovlp_i = np.einsum('peij,peij->peij', d_shift_perm_i, d_cs_ovlp_i)
    d2_cs_ovlp_i -= np.einsum("ei,pi,peij->peij", np.eye(dd1_obj.ham.N), shift.imag, d_cs_ovlp_i)
    d2_cs_ovlp_i += 1j * 0.5 * np.einsum("pb,ej,jb,peij->peij", shift, perm_mat.T, perm_mat, d_cs_ovlp_i, optimize=True)
    d2_cs_ovlp_i -= 1j * 0.5 * np.einsum("pb,bj,je,peij->peij", shift.conj(), perm_mat.T, perm_mat, d_cs_ovlp_i, optimize=True)
    d2_cs_ovlp_i -= np.einsum('ei,ij,p->peij', np.eye(dd1_obj.ham.N), cs_ovlp, np.ones(dd1_obj.ham.N))
    d2_cs_ovlp_i -= np.einsum('kp,ej,ij->peij',perm_mat, perm_mat.T, cs_ovlp)
    d2_cs_ovlp_i += 2 * np.einsum('ei,pp,ej,ij->peij', np.eye(dd1_obj.ham.N), perm_mat, perm_mat.T, cs_ovlp)

    elph_ij_contracted = np.einsum("ijk,ij->ijk", dd1_obj.ham.g_tensor, Ga_i)
    g_contracted = np.einsum("ijk,ki->ij", dd1_obj.ham.g_tensor, shift.conj())
    g_contracted += np.einsum("ijk,kj->ij", dd1_obj.ham.g_tensor, beta_i)
    g_contracted = g_contracted * cs_ovlp
    g_contracted = g_contracted.dot(perm_mat)

    # Shift Hess Real
    elph_contrib = np.einsum('ejp,peej->pe', elph_ij_contracted, d_cs_ovlp_r) + np.einsum('ijk,kp,ej,peij->pe', elph_ij_contracted, perm_mat, perm_mat.T, d_cs_ovlp_r) #flip T
    elph_contrib *= 2.
    elph_contrib += np.einsum('ijk,ki,peij->pe', elph_ij_contracted, shift.conj(), d2_cs_ovlp_r) + np.einsum('ijk,kj,peij->pe', elph_ij_contracted, beta_i, d2_cs_ovlp_r)
    elph_tmp = [elph_contrib.real + 0j]

    # Shift Hess Imag
    elph_contrib = -1j * np.einsum('ejp,peej->pe', elph_ij_contracted, d_cs_ovlp_i) + 1j * np.einsum('ijk,kp,ej,peij->pe', elph_ij_contracted, perm_mat, perm_mat.T, d_cs_ovlp_i)
    elph_contrib *= 2.
    elph_contrib += np.einsum('ijk,ki,peij->pe', elph_ij_contracted, shift.conj(), d2_cs_ovlp_i) + np.einsum('ijk,kj,peij->pe', elph_ij_contracted, beta_i, d2_cs_ovlp_i)
    elph_tmp[0] += 1j * elph_contrib.real
    
    # Elec Hess Real
    elph_contrib = 2 * g_contracted.diagonal()
    elph_tmp.append(elph_contrib.real + 0j)

    # Elec Hess Imag
    elph_contrib = 2 * g_contracted.diagonal()
    elph_tmp[1] += 1j * elph_contrib.real

    elph_x = dd1_obj.pack_x(elph_tmp[0].ravel(), elph_tmp[1].ravel())
    elph_en_hess = (fac_i * elph_x).real

    # PACKING SEEMS TRICKY, AS OVLP CONTRIBS MAY BE COMPLEX VALUED, SO PARTS WILL BE REDISTRIBUTED IN x.
    return elph_en_hess

def analytical_hess_elph_tot(x, dd1_obj):
    elph_en_hess = np.zeros_like(x, dtype=np.float64)
    for iperm in range(dd1_obj.nperms):
        elph_en_hess += analytical_hess_elph(x, dd1_obj, iperm)
    return elph_en_hess

def overlap(x: np.ndarray, dd1_obj, iperm) -> float:
    shift, c0a, c0b = dd1_obj.unpack_x(x)
    shift = np.squeeze(shift)
    c0a = np.squeeze(c0a)
    shift_i = npj.roll(shift, shift=(-iperm, -iperm), axis=(0, 1))
    fac_i = overlap_degeneracy(dd1_obj.ham, iperm) * dd1_obj.Kcoeffs[0].conj() * dd1_obj.Kcoeffs[iperm]

    cs_ovlp = dd1_obj.cs_overlap(shift, shift_i)
    return (fac_i * npj.einsum("i,i,i->", c0a.conj(), c0a[dd1_obj.perms[iperm]], cs_ovlp.diagonal())).real     

def overlap_tot(x: np.ndarray, dd1_obj):
    ovlp = 0.
    for iperm in range(dd1_obj.nperms):
        ovlp += overlap(x, dd1_obj, iperm)
    return ovlp

def analytical_hess_overlap(x, dd1_obj, iperm):
    # Def Ga_i, d2_cs_ovlp_r, d2_cs_ovlp_i, ovlp_contracted, fac_i
    shift, c0a, c0b = dd1_obj.unpack_x(x)
    shift = np.squeeze(shift)
    c0a = np.squeeze(c0a)
    beta_i = np.roll(shift, shift=(-iperm, -iperm), axis=(0, 1))
    psia_i = c0a[dd1_obj.perms[iperm]]

    perm_mat = np.roll(np.eye(dd1_obj.ham.N), shift=-iperm, axis=0) 
    cs_ovlp = dd1_obj.cs_overlap(shift, beta_i)   
    fac_i = overlap_degeneracy(dd1_obj.ham, iperm) * dd1_obj.Kcoeffs[0].conj() * dd1_obj.Kcoeffs[iperm]
    Ga_i = np.outer(c0a.conj(), psia_i)
    ovlp_contracted = np.einsum("i,ij->ij", cs_ovlp.diagonal(), perm_mat)

    d_shift_perm_r = np.einsum("ei,pj->peij", np.eye(dd1_obj.ham.N), perm_mat.dot(shift).dot(perm_mat.T))
    d_shift_perm_r += np.einsum("pi,je->peij", perm_mat.T.dot(shift.conj()), perm_mat)

    d_cs_ovlp_r = np.einsum("peij,ij->peij", d_shift_perm_r, cs_ovlp)
    d_cs_ovlp_r -= np.einsum("ei,pi,ij->peij", np.eye(dd1_obj.ham.N), shift.real, cs_ovlp)
    d_cs_ovlp_r -= np.einsum("kp,ej,kj,ij->peij", perm_mat, perm_mat.T, beta_i.real, cs_ovlp) 

    d2_cs_ovlp_r = np.einsum('peij,peij->peij', d_shift_perm_r, d_cs_ovlp_r)
    d2_cs_ovlp_r -= np.einsum("ei,pi,peij->peij", np.eye(dd1_obj.ham.N), shift.real, d_cs_ovlp_r)
    d2_cs_ovlp_r -= np.einsum("kp,ej,kj,peij->peij", perm_mat, perm_mat.T, beta_i.real, d_cs_ovlp_r) #do these need 1/2?
    d2_cs_ovlp_r -= np.einsum('ei,ij,p->peij', np.eye(dd1_obj.ham.N), cs_ovlp, np.ones(dd1_obj.ham.N))
    d2_cs_ovlp_r -= np.einsum('kp,ej,ij->peij',perm_mat, perm_mat.T, cs_ovlp)
    d2_cs_ovlp_r += 2 * np.einsum('ei,pp,ej,ij->peij', np.eye(dd1_obj.ham.N), perm_mat, perm_mat.T, cs_ovlp)
    
    d_shift_perm_i = -1j * np.einsum("ei,pj->peij", np.eye(dd1_obj.ham.N), perm_mat.dot(shift).dot(perm_mat.T))
    d_shift_perm_i += 1j * np.einsum("pi,je->peij", perm_mat.T.dot(shift.conj()), perm_mat)
    
    d_cs_ovlp_i = np.einsum("peij,ij->peij", d_shift_perm_i, cs_ovlp)
    d_cs_ovlp_i -= np.einsum("ei,pi,ij->peij", np.eye(dd1_obj.ham.N), shift.imag, cs_ovlp)
    d_cs_ovlp_i -=  np.einsum("pb,ej,jb,ij->peij", shift.imag, perm_mat.T, perm_mat, cs_ovlp, optimize=True)

    d2_cs_ovlp_i = np.einsum('peij,peij->peij', d_shift_perm_i, d_cs_ovlp_i)
    d2_cs_ovlp_i -= np.einsum("ei,pi,peij->peij", np.eye(dd1_obj.ham.N), shift.imag, d_cs_ovlp_i)
    d2_cs_ovlp_i += 1j * 0.5 * np.einsum("pb,ej,jb,peij->peij", shift, perm_mat.T, perm_mat, d_cs_ovlp_i, optimize=True)
    d2_cs_ovlp_i -= 1j * 0.5 * np.einsum("pb,bj,je,peij->peij", shift.conj(), perm_mat.T, perm_mat, d_cs_ovlp_i, optimize=True)
    d2_cs_ovlp_i -= np.einsum('ei,ij,p->peij', np.eye(dd1_obj.ham.N), cs_ovlp, np.ones(dd1_obj.ham.N))
    d2_cs_ovlp_i -= np.einsum('kp,ej,ij->peij',perm_mat, perm_mat.T, cs_ovlp)
    d2_cs_ovlp_i += 2 * np.einsum('ei,pp,ej,ij->peij', np.eye(dd1_obj.ham.N), perm_mat, perm_mat.T, cs_ovlp)

    # Shift Hess Real 
    ovlp_contrib = np.einsum("i,peii->pe", Ga_i.diagonal(), d2_cs_ovlp_r) 
#    ovlp_tmp = [ovlp_contrib.real + 0j] 
    ovlp_tmp = ovlp_contrib.ravel()

    # Shift Hess Imag
    ovlp_contrib = np.einsum("i,peii->pe", Ga_i.diagonal(), d2_cs_ovlp_i)
#    ovlp_tmp[0] += 1j * ovlp_contrib.real
    ovlp_tmp = np.hstack([ovlp_tmp, ovlp_contrib.ravel()])

    # Elec Hess Real
    ovlp_contrib = 2 * ovlp_contracted.diagonal()
#    ovlp_tmp.append(ovlp_contrib.real + 0j)
    ovlp_tmp = np.hstack([ovlp_tmp, ovlp_contrib.ravel()])

    # Elec Hess Imag
    ovlp_contrib = 2 * ovlp_contracted.diagonal()
#    ovlp_tmp[1] += 1j * ovlp_contrib.real
    ovlp_x = np.hstack([ovlp_tmp, ovlp_contrib.ravel()]) 

#    ovlp_x = dd1_obj.pack_x(ovlp_tmp[0].ravel(), ovlp_tmp[1].ravel())
    ovlp_en_hess = (fac_i * ovlp_x).real

    # PACKING SEEMS TRICKY, AS OVLP CONTRIBS MAY BE COMPLEX VALUED, SO PARTS WILL BE REDISTRIBUTED IN x.
    return ovlp_en_hess

def analytical_hess_overlap_tot(x, dd1_obj):
    ovlp_en_hess = np.zeros_like(x, dtype=np.float64)
    for iperm in range(dd1_obj.nperms):
        ovlp_en_hess += analytical_hess_overlap(x, dd1_obj, iperm)

    return ovlp_en_hess

def test_ovlp_hess(x, dd1_obj, iperm):
    analytical_hess = analytical_hess_overlap(x, dd1_obj, iperm)
    jax_hess = jax.hessian(overlap)(x, dd1_obj, iperm).diagonal()
    print(np.max(np.abs(analytical_hess - jax_hess)))

def test_ovlp_hess_tot(x, dd1_obj):
    for iperm in range(dd1_obj.nperms):
        anal_hess_iperm = analytical_hess_overlap(x, dd1_obj, iperm)
        jax_hess_iperm = jax.hessian(overlap)(x, dd1_obj, iperm).diagonal()
        print(f'analytical hess {iperm} diff: ', anal_hess_iperm - jax_hess_iperm)
        print(f'max diff {iperm}:   ', np.max(np.abs(anal_hess_iperm - jax_hess_iperm)))

    analytical_hess_tot = analytical_hess_overlap_tot(x, dd1_obj)
    jax_hess_tot = jax.hessian(overlap_tot)(x, dd1_obj).diagonal()
    print('analytical hess tot diff: ', analytical_hess_tot - jax_hess_tot)
    print(np.max(np.abs(analytical_hess_tot - jax_hess_tot)))

def test_elph_hess(x, dd1_obj, iperm):
    analytical_hess = analytical_hess_elph(x, dd1_obj, iperm)
    jax_hess = jax.hessian(elph)(x, dd1_obj, iperm).diagonal()
    print(np.max(np.abs(analytical_hess - jax_hess)))

def test_elph_hess_tot(x, dd1_obj):
    for iperm in range(dd1_obj.nperms):
        anal_hess_iperm = analytical_hess_elph(x, dd1_obj, iperm)
        jax_hess_iperm = jax.hessian(elph)(x, dd1_obj, iperm).diagonal()
        print(f'analytical hess {iperm} diff: ', anal_hess_iperm - jax_hess_iperm)
        print(f'max diff {iperm}:   ', np.max(np.abs(anal_hess_iperm - jax_hess_iperm)))

    analytical_hess_tot = analytical_hess_elph_tot(x, dd1_obj)
    jax_hess_tot = jax.hessian(elph_tot)(x, dd1_obj).diagonal()
    print('analytical hess tot diff: ', analytical_hess_tot - jax_hess_tot)
    print(np.max(np.abs(analytical_hess_tot - jax_hess_tot)))

def test_bos_hess(x, dd1_obj, iperm):
    analytical_hess = analytical_hess_bos(x, dd1_obj, iperm)
    jax_hess = jax.hessian(bos)(x, dd1_obj, iperm).diagonal()
    print(np.max(np.abs(analytical_hess - jax_hess)))

def test_bos_hess_tot(x, dd1_obj):
    for iperm in range(dd1_obj.nperms):
        anal_hess_iperm = analytical_hess_bos(x, dd1_obj, iperm)
        jax_hess_iperm = jax.hessian(boson)(x, dd1_obj, iperm).diagonal()
        print(f'analytical hess {iperm} diff: ', anal_hess_iperm - jax_hess_iperm)
        print(f'max diff {iperm}:   ', np.max(np.abs(anal_hess_iperm - jax_hess_iperm)))

    analytical_hess_tot = analytical_hess_bos_tot(x, dd1_obj)
    jax_hess_tot = jax.hessian(boson_tot)(x, dd1_obj).diagonal()
    print('analytical hess tot diff: ', analytical_hess_tot - jax_hess_tot)
    print(np.max(np.abs(analytical_hess_tot - jax_hess_tot)))

def main():
    # System Parameters
    nup = 1
    ndown = 0
    nelec = (nup, ndown)

    # Hamiltonian Parameters
    g = 0.7
    t = 1.
    w0 = 0.5
    nsites = 10
    pbc = True

    system = Generic(nelec)
    ham = OpticalSSHModel(g=g, t=t, w0=w0, nsites=nsites, pbc=pbc)
    ham.build()

    initial_electron = np.random.random((nsites, nup + ndown)) + 1j * np.random.random((nsites, nup + ndown))
    initial_phonons = np.random.normal(size=(nsites, nsites)) + 1j * np.random.normal(size=(nsites, nsites))

    dd1_obj = dD1Variational(initial_phonons, initial_electron, ham, system, cplx=True, K=0.) 
    x = dd1_obj.pack_x()

    test_ovlp_hess_tot(x, dd1_obj)
#    test_elph_hess_tot(x, dd1_obj)
#    test_bos_hess_tot(x, dd1_obj)

if __name__ == '__main__':
    main()

