from src.qgt import *


def extractMinimalHawkingPartner(bogoliubovTransformation, modeIndex, atol=1e-8, partnerCriterion='B1'):
    """
    Builds a reduced 4x4 Bogoliubov transformation for the Hawking mode and its partner,
    without extending to a full orthonormal basis or using full 2N×2N matrices.

    Parameters:
        bogoliubovTransformation: full 2N x 2N Bogoliubov transformation matrix
        modeIndex: index of selected Hawking mode (in 0..N-1)
        atol: numerical tolerance
        partnerCriterion: 'B1' (default) or 'B2', selects which criterion to use to define the partner mode

    Returns:
        newBogoliubovTransformation_4x4: Bogoliubov transformation acting only on Hawking+partner
    """

    alphaVec, betaVec = getAlphaBetaVectors(bogoliubovTransformation, modeIndex)

    # Compute decomposition of the Hawking mode
    alpha, betaParallelConj, betaPerpConj = computeBetaParallelAndPerp(alphaVec, betaVec, atol=atol)
    betaParallel = np.conj(betaParallelConj)
    betaPerp = np.conj(betaPerpConj)

    # Choose partner mode coefficients according to selected criterion
    if partnerCriterion == 'B1':
        gp, gpp, dp, dpp = computePartnerMode(alpha, betaParallelConj, betaPerpConj, atol=atol)
    elif partnerCriterion == 'B2':
        gp, gpp, dp, dpp = computePartnerB2Coefficients(alpha, betaParallel, betaPerp, atol=atol)
    else:
        raise ValueError(f"Unknown partnerCriterion: {partnerCriterion}. Use 'B1' or 'B2'.")

    # Build reduced 4x4 transformation from IN to Hawking/Partner basis
    reducedTransformation = np.array([
        [np.conj(alpha), -betaParallelConj, 0, -betaPerpConj],
        [-betaParallel, alpha, -betaPerp, 0],
        [np.conj(gp), -np.conj(dp), np.conj(gpp), -np.conj(dpp)],
        [-dp, gp, -dpp, gpp]
    ], dtype=complex)

    return reducedTransformation



def getAlphaBetaVectors(S, modeIndex):
    """
    Extracts the alpha and beta vectors from the matrix S for a given mode.

    Parameters:
        S: numpy matrix of dimension (2n, 2n), can be complex
        modeIndex: integer (index of the mode, starting from 0)

    Returns:
        alphaVector: elements at even positions of row 2*modeIndex, conjugated
        betaVector: elements at odd positions of row 2*modeIndex, as they are
    """
    row = S[2 * modeIndex]
    alphaVector = np.conj(row[::2])  # elements 0, 2, 4, ... conjugated
    betaVector = np.conj(-row[1::2])  # elements 1, 3, 5, ... not conjugated
    return alphaVector, betaVector


def computeBetaParallelAndPerp(alphaVec, betaVec, atol=1e-8):
    """
    Given an alphaVec and betaVec, returns:
        - alpha: COMPLEX scalar component of alphaVec on nParallel
        - betaParallel: COMPLEX scalar component of betaVec on nParallel
        - betaPerp: COMPLEX scalar component of betaVec on nPerp
    """

    alphaNorm = np.linalg.norm(alphaVec)
    if alphaNorm < atol:
        raise ValueError("The alpha vector cannot be null.")

    # Parallel vector (Hawking mode)
    nParallel = alphaVec / alphaNorm
    alpha = np.vdot(nParallel, alphaVec)
    betaConj = np.conj(betaVec)
    betaParallel_conj = np.vdot(nParallel, betaConj)

    # Perpendicular vector
    betaResidual = betaConj - betaParallel_conj * nParallel
    betaPerpNorm = np.linalg.norm(betaResidual)

    if betaPerpNorm > atol:
        nPerp = betaResidual / betaPerpNorm
        betaPerp_conj = np.vdot(nPerp, betaConj)
    else:
        betaPerp_conj = 0.0

    # Check commutator
    commutator = abs(alpha) ** 2 - abs(betaParallel_conj) ** 2 - abs(betaPerp_conj) ** 2
    if abs(commutator - 1) > 1e-6:
        raise ValueError(f"Commutator is not 1: {commutator}")

    return alpha, betaParallel_conj, betaPerp_conj


def computePartnerMode(alpha, betaParallelConj, betaPerpConj, atol=1e-8):
    """
    Given a mode a_H = alpha * a_|| + betaParallel * a_||^† + betaPerp * a_perp^†,
    returns the coefficients of the partner mode:

    a_p = gammaParallel^* a_|| + gammaPerp^* a_perp +
          deltaParallel a_||^† + deltaPerp a_perp^†

    All coefficients are complex scalars.
    """
    betaParallel = np.conj(betaParallelConj)
    betaPerp = np.conj(betaPerpConj)

    if np.abs(betaParallel) < atol:
        # Special case: betaParallel = 0
        gammaParallel = 0.0 + 1j * 0
        gammaPerp = alpha
        deltaParallel = betaPerp
        deltaPerp = 0.0 + 1j * 0

    elif np.abs(betaPerp) < atol:
        # Special case: betaPerp = 0 (no partner needed, it is itself)
        gammaParallel = alpha
        gammaPerp = 0
        deltaParallel = 0
        deltaPerp = 0

        print("No partner needed. Some calculations may fail from now on")
    else:
        # Option B1 from paper 1503.06109 modified

        # Set deltaPerp = 0
        deltaPerp = 0.0 + 1j * 0

        # Define factors
        A = betaParallelConj / np.conj(alpha)
        B = (alpha - A * betaParallel) / betaPerp

        # Solve the equation:
        # |A * d|^2 + |B * d|^2 - |d|^2 = 1
        # => |d|^2 * (|A|^2 + |B|^2 - 1) = 1
        normFactor = np.abs(A) ** 2 + np.abs(B) ** 2 - 1

        if np.abs(normFactor) < atol:
            raise ValueError("Norm condition leads to divergence (denominator ~ 0)")

        # Absolute value of deltaParallel is easily obtained
        absDeltaParallel = np.sqrt(1 / normFactor)
        deltaParallel = absDeltaParallel  # We choose deltaParallel to be real by choosing the phase

        # Now calculate the gamma coefficients
        gammaParallel = A * deltaParallel
        gammaPerp = B * deltaParallel

    conmutatorPartner = np.abs(gammaParallel) ** 2 + np.abs(gammaPerp) ** 2 - np.abs(deltaParallel) ** 2 - np.abs(
        deltaPerp) ** 2

    hawkingPartnerConmutator = np.conj(gammaParallel) * alpha - betaParallel * np.conj(
        deltaParallel) - betaPerp * np.conj(deltaPerp)

    partnerHawkingConmutator = np.conj(gammaParallel) * betaParallelConj + np.conj(gammaPerp) * betaPerpConj - np.conj(
        alpha) * np.conj(deltaParallel)

    if abs(conmutatorPartner - 1.0) > 1e-4:
        raise ValueError("Partner commutation relation fails to be fulfilled")

    if abs(hawkingPartnerConmutator) > 1e-4 or abs(partnerHawkingConmutator) > 1e-4:
        raise ValueError("Hawking partner commutation fails to be fulfilled")

    return gammaParallel, gammaPerp, deltaParallel, deltaPerp


def computePartnerB2Coefficients(alpha, betaParallel, betaPerp, atol=1e-8):
    """
    Given the decomposition of a Hawking mode:
        b_H = alpha * a_parallel + betaParallel* a_parallel† + betaPerp* a_perp†,
    computes the partner mode coefficients under criterion B2:
        b_P = gammaParallel * a_parallel + gammaPerp * a_perp + 
              deltaParallel * a_parallel† + deltaPerp * a_perp†
    """

    # Criterio B2: gamma ∝ conjugate(beta), delta ∝ conjugate(alpha)
    # deltaPerp se pone a cero según el criterio
    deltaPerp = 0.0

    # Supón que: gamma_parallel = k * conjugate(beta_parallel)
    #            gamma_perp = k * conjugate(beta_perp)
    #            delta_parallel = k * conjugate(alpha)
    # Busca k tal que se cumpla la normalización:
    # |gamma_parallel|^2 + |gamma_perp|^2 - |delta_parallel|^2 = 1

    betaNormSq = abs(betaParallel)**2 + abs(betaPerp)**2
    alphaNormSq = abs(alpha)**2

    denominator = betaNormSq - alphaNormSq

    if np.abs(denominator) < atol:
        raise ValueError("Cannot normalize partner mode: denominator too small.")

    kSquared = 1.0 / denominator
    if kSquared.real < 0:
        raise ValueError("No real positive normalization possible under B2.")

    k = np.sqrt(kSquared)

    gammaParallel = k * np.conj(betaParallel)
    gammaPerp = k * np.conj(betaPerp)
    deltaParallel = k * np.conj(alpha)

    # Verificación del conmutador:
    commutator = (
        abs(gammaParallel)**2 + abs(gammaPerp)**2
        - abs(deltaParallel)**2 - abs(deltaPerp)**2
    )

    if np.abs(commutator - 1) > 1e-6:
        raise ValueError(f"Commutator [bP, bP†] != 1: got {commutator}")

    return gammaParallel, gammaPerp, deltaParallel, deltaPerp

