# nested_interpolation.py

import taichi as ti

import taichi.math as tm

@ti.func
def compute_cumulants(f: ti.template(), I): # f[I] is source
    f0 = f[I][0]
    f1 = f[I][1]
    f2 = f[I][2]
    f3 = f[I][3]
    f4 = f[I][4]
    f5 = f[I][5]
    f6 = f[I][6]
    f7 = f[I][7]
    f8 = f[I][8]
    x0 = f1 + f8
    x1 = f3 + f7 + x0
    x2 = f5 + f6
    x3 = f2 + x2
    x4 = f4 + x3
    x5 = -f7
    x6 = f5 - f6
    x7 = -f8
    m00 = f0 + x1 + x4
    m10 = -f3 + x0 + x5 + x6
    m01 = -f4 + x3 + x5 + x7
    m20 = x1 + x2
    m02 = f7 + f8 + x4
    m11 = f7 + x6 + x7
    rho = m00
    u = m10/rho
    v = m01/rho
    c20 = m20/rho - u**2
    c02 = m02/rho - v**2
    c11 = m11/rho - u*v
    return rho, u, v, c20, c02, c11


Coeffs    = ti.types.struct(C0=float, Cx=float, Cy=float, Cxy=float, Cx2=float, Cy2=float)

RhoCoeffs = ti.types.struct(C0=float, Cx=float, Cy=float, Cxy=float)

@ti.func
def compute_interpolation_coeffs(f: ti.template(), I, omega):
    Imm = ti.Vector([I[0]  , I[1]  ])
    Ipm = ti.Vector([I[0]+1, I[1]  ])
    Ipp = ti.Vector([I[0]+1, I[1]+1])
    Imp = ti.Vector([I[0]  , I[1]+1])
    rho_mm, u_mm, v_mm, c20_mm, c02_mm, c11_mm = compute_cumulants(f, Imm)
    rho_pm, u_pm, v_pm, c20_pm, c02_pm, c11_pm = compute_cumulants(f, Ipm)
    rho_pp, u_pp, v_pp, c20_pp, c02_pp, c11_pp = compute_cumulants(f, Ipp)
    rho_mp, u_mp, v_mp, c20_mp, c02_mp, c11_mp = compute_cumulants(f, Imp)
    x0 = 3*omega/16
    x1 = c11_pm*x0
    x2 = 3*omega/32
    x3 = c02_pp*x2
    x4 = c20_mm*x2
    x5 = c11_mp*x0
    x6 = c02_mm*x2
    x7 = c20_pp*x2
    x8 = c02_mp*x2 - c02_pm*x2 - c11_mm*x0 + c11_pp*x0 - c20_mp*x2 + c20_pm*x2
    x9 = u_mp/2
    x10 = u_pm/2
    x11 = -x10
    x12 = u_mm/2
    x13 = u_pp/2
    x14 = -x13
    x15 = x12 + x14
    x16 = -x9
    x17 = 3*omega/8
    x18 = c02_mp*x17
    x19 = c20_pm*x17
    x20 = v_mp/2
    x21 = -v_pp/2
    x22 = v_pm/2
    x23 = v_mm/2
    x24 = x20 + x21 + x22 - x23
    x25 = c02_mm*x17 - c02_pp*x17 - c20_mm*x17 + c20_pp*x17
    x26 = 3*omega/4
    x27 = c11_mp*x26
    x28 = c11_pm*x26
    x29 = c11_mm*x26 - c11_pp*x26
    x30 = x21 + x23
    x31 = rho_mp/2
    x32 = rho_pm/2
    x33 = rho_mm/2 - rho_pp/2
    u_coeffs = Coeffs(
        C0 = u_mm/4 + u_mp/4 + u_pm/4 + u_pp/4 - x1 - x3 - x4 + x5 + x6 + x7 + x8,
        Cx = -x11 - x15 - x9,
        Cy = -x10 - x15 - x16,
        Cxy = u_mm - u_mp - u_pm + u_pp,
        Cx2 = 3*c02_pm*omega/8 + 3*c20_mp*omega/8 - x18 - x19 - x24 - x25,
        Cy2 = x24 - x27 + x28 + x29,
    )
    v_coeffs = Coeffs(
        C0 = v_mm/4 + v_mp/4 + v_pm/4 + v_pp/4 + x1 + x3 + x4 - x5 - x6 - x7 + x8,
        Cx = -x20 + x22 - x30,
        Cy = x20 - x22 - x30,
        Cxy = v_mm - v_mp - v_pm + v_pp,
        Cx2 = x10 - x12 + x14 + x27 - x28 + x29 + x9,
        Cy2 = c02_pm*x17 + c20_mp*x17 + x11 + x12 + x13 + x16 - x18 - x19 + x25,
    )
    rho_coeffs = RhoCoeffs(
        C0 = rho_mm/4 + rho_mp/4 + rho_pm/4 + rho_pp/4,
        Cx = -x31 + x32 - x33,
        Cy = x31 - x32 - x33,
        Cxy = rho_mm - rho_mp - rho_pm + rho_pp,
    )
    return u_coeffs, v_coeffs, rho_coeffs


@ti.func
def backtrans_c_to_f(f: ti.template(), I, rho, u, v, c20, c02, c11): # f[I] is injection target: F->C or C->F
    kappa20 = c20*rho
    kappa02 = c02*rho
    kappa11 = c11*rho
    kappa21 = 0
    kappa12 = 0
    kappa22 = (kappa02*kappa20 + 2*kappa11**2)/rho
    m_post_idx_0_c_0 = rho
    m_post_idx_0_c_1 = rho*v
    m_post_idx_0_c_2 = kappa02 + rho*v**2
    m_post_idx_1_c_0 = 0
    m_post_idx_1_c_1 = kappa11
    m_post_idx_1_c_2 = 2*kappa11*v + kappa12
    m_post_idx_2_c_0 = kappa20
    m_post_idx_2_c_1 = kappa20*v + kappa21
    m_post_idx_2_c_2 = kappa20*v**2 + 2*kappa21*v + kappa22
    m00_post = m_post_idx_0_c_0
    m01_post = m_post_idx_0_c_1
    m02_post = m_post_idx_0_c_2
    m10_post = m_post_idx_0_c_0*u + m_post_idx_1_c_0
    m11_post = m_post_idx_0_c_1*u + m_post_idx_1_c_1
    m12_post = m_post_idx_0_c_2*u + m_post_idx_1_c_2
    m20_post = m_post_idx_0_c_0*u**2 + 2*m_post_idx_1_c_0*u + m_post_idx_2_c_0
    m21_post = m_post_idx_0_c_1*u**2 + 2*m_post_idx_1_c_1*u + m_post_idx_2_c_1
    m22_post = m_post_idx_0_c_2*u**2 + 2*m_post_idx_1_c_2*u + m_post_idx_2_c_2
    chimera_m_post_0_0 = m00_post - m20_post
    chimera_m_post_m1_0 = -m10_post/2 + m20_post/2
    chimera_m_post_1_0 = m10_post/2 + m20_post/2
    chimera_m_post_0_1 = m01_post - m21_post
    chimera_m_post_m1_1 = -m11_post/2 + m21_post/2
    chimera_m_post_1_1 = m11_post/2 + m21_post/2
    chimera_m_post_0_2 = m02_post - m22_post
    chimera_m_post_m1_2 = -m12_post/2 + m22_post/2
    chimera_m_post_1_2 = m12_post/2 + m22_post/2
    f[I][0] = chimera_m_post_0_0 - chimera_m_post_0_2
    f[I][1] = chimera_m_post_1_0 - chimera_m_post_1_2
    f[I][2] = chimera_m_post_0_1/2 + chimera_m_post_0_2/2
    f[I][3] = chimera_m_post_m1_0 - chimera_m_post_m1_2
    f[I][4] = -chimera_m_post_0_1/2 + chimera_m_post_0_2/2
    f[I][5] = chimera_m_post_1_1/2 + chimera_m_post_1_2/2
    f[I][6] = chimera_m_post_m1_1/2 + chimera_m_post_m1_2/2
    f[I][7] = -chimera_m_post_m1_1/2 + chimera_m_post_m1_2/2
    f[I][8] = -chimera_m_post_1_1/2 + chimera_m_post_1_2/2


@ti.func
def interpolation_FtoC(UC, VC, RC, omega):
    INV_3omega = 1/(3*omega)
    UpScaling=2 # reconstructed gradient at F becomes twice when C observes it (chain rule in x_F <-> x_C)
    u = UC.C0
    v = VC.C0
    rho = RC.C0
    c11 = -(UC.Cy + VC.Cx) * INV_3omega * UpScaling
    c20 = -2*UC.Cx * INV_3omega * UpScaling + 1/3
    c02 = -2*VC.Cy * INV_3omega * UpScaling + 1/3
    return u, v, rho, c20, c02, c11


@ti.func
def interpolation_CtoF(UC, VC, RC, omega, x, y):
    INV_3omega = 1/(3*omega)
    DownScaling=0.5 # reconstructed gradient at C becomes half  when F observes it (chain rule in x_C <-> x_F)
    u = UC.C0 + UC.Cx*x + UC.Cx2*x*x + UC.Cxy*x*y + UC.Cy*y + UC.Cy2*y*y
    v = VC.C0 + VC.Cx*x + VC.Cx2*x*x + VC.Cxy*x*y + VC.Cy*y + VC.Cy2*y*y
    rho = RC.C0 + RC.Cx*x + RC.Cxy*x*y + RC.Cy*y
    c20 = -2*(UC.Cx + 2*UC.Cx2*x + UC.Cxy*y) * INV_3omega * DownScaling + 1/3
    c02 = -2*(VC.Cxy*x + VC.Cy + 2*VC.Cy2*y) * INV_3omega * DownScaling + 1/3
    c11 = -(UC.Cxy*x + UC.Cy + 2*UC.Cy2*y + VC.Cx + 2*VC.Cx2*x + VC.Cxy*y) * INV_3omega * DownScaling
    return u, v, rho, c20, c02, c11
