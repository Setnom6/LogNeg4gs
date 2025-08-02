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
    nParallel, alpha, nPerp, betaParallel, betaPerp = computeBetaParallelAndPerp(alphaVec, betaVec, atol=atol)

    # Choose partner mode coefficients according to selected criterion
    if partnerCriterion == 'B1':
        gp, gpp, dp, dpp = computePartnerMode(alpha, betaParallel, betaPerp, atol=atol)
    elif partnerCriterion == 'B2':
        gp, gpp, dp, dpp = computePartnerB2Coefficients(alpha, betaParallel, betaPerp, atol=atol)
    else:
        raise ValueError(f"Unknown partnerCriterion: {partnerCriterion}. Use 'B1' or 'B2'.")

    # Build reduced 4x4 transformation from IN to Hawking/Partner basis
    reducedTransformation = np.array([
        [alpha, np.conj(betaParallel), 0, np.conj(betaPerp)],
        [betaParallel, np.conj(alpha), betaPerp, 0],
        [gp, np.conj(dp), gpp, np.conj(dpp)],
        [dp, np.conj(gp), dpp, np.conj(gpp)]
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
    alphaVector = row[::2]  # elements 0, 2, 4, ... conjugated
    betaVector = np.conj(row[1::2])  # elements 1, 3, 5, ... not conjugated
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
    betaParallel = np.vdot(nParallel, betaVec)

    # Perpendicular vector
    betaResidual = betaVec - betaParallel * nParallel
    betaPerpNorm = np.linalg.norm(betaResidual)

    if betaPerpNorm > atol:
        nPerp = betaResidual / betaPerpNorm
        betaPerp = np.vdot(nPerp, betaVec)
    else:
        nPerp = np.zeros_like(betaVec)
        betaPerp = 0.0

    # Check commutator
    commutator = abs(alpha) ** 2 - abs(betaParallel) ** 2 - abs(betaPerp) ** 2

    if abs(commutator - 1) > 1e-6:
        raise ValueError(f"Commutator is not 1: {commutator}")

    return nParallel, alpha, nPerp, betaParallel, betaPerp


def computePartnerMode(alpha, betaParallel, betaPerp, atol=1e-8):
    """
    Given a mode a_H = alpha * a_|| + betaParallel * a_||^† + betaPerp * a_perp^†,
    returns the coefficients of the partner mode:

    a_p = gammaParallel^* a_|| + gammaPerp^* a_perp +
          deltaParallel a_||^† + deltaPerp a_perp^†

    All coefficients are complex scalars. It assumes the criterion B1 where deltaPerp = 0.0
    """
    betaParallelConj = np.conj(betaParallel)
    betaPerpConj = np.conj(betaPerp)
    alphaConj = np.conj(alpha)

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
        A = betaParallel / alphaConj
        B = (alpha - A * betaParallelConj) / betaPerpConj

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

    hawkingPartnerConmutator = alpha * np.conj(
        deltaParallel) - betaParallelConj * gammaParallel - betaPerpConj * gammaPerp

    HawkingDaggerPartnerConmutator = alphaConj * gammaParallel - betaParallel * np.conj(
        deltaParallel) - betaPerp * np.conj(deltaPerp)

    if abs(conmutatorPartner - 1.0) > 1e-4:
        raise ValueError("Partner commutation relation fails to be fulfilled")

    if abs(hawkingPartnerConmutator) > 1e-4:
        raise ValueError("Hawking partner commutation fails to be fulfilled")

    if abs(HawkingDaggerPartnerConmutator) > 1e-4:
        raise ValueError("Hawking dagger partner commutation fails to be fulfilled")

    return gammaParallel, gammaPerp, deltaParallel, deltaPerp


def computePartnerB2Coefficients(alpha, betaParallel, betaPerp, atol=1e-8):
    """
    Given a mode a_H = alpha * a_|| + betaParallel * a_||^† + betaPerp * a_perp^†,
    returns the coefficients of the partner mode:

    a_p = gammaParallel^* a_|| + gammaPerp^* a_perp +
          deltaParallel a_||^† + deltaPerp a_perp^†

    All coefficients are complex scalars. It assumes the criterion B1 where gamma is parallel to beta.
    """

    betaParallelConj = np.conj(betaParallel)
    betaPerpConj = np.conj(betaPerp)

    if np.abs(betaPerp) < atol:
        # Special case: betaPerp = 0 (no partner needed, it is itself)
        gammaParallel = alpha
        gammaPerp = 0
        deltaParallel = 0
        deltaPerp = 0

    else:

        # Option B2 from paper 1503.06109 modified

        # Now compute delta coeffs without lambda proportionality

        A = (betaParallelConj * betaParallel + betaPerpConj * betaPerp) / np.conj(alpha)
        B = (betaParallelConj * alpha - betaParallelConj * A) / betaPerpConj

        # Use conmutation relation of partner to set lambda (we assume it to be real)
        lambdaCoeff = 1.0 / (
                betaParallel * betaParallelConj + betaPerp * betaPerpConj - np.abs(A) ** 2 - np.abs(B) ** 2)

        gammaParallel = betaParallel * lambdaCoeff
        gammaPerp = betaPerp * lambdaCoeff
        deltaParallel = A * np.conj(lambdaCoeff)
        deltaPerp = B * np.conj(lambdaCoeff)

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
