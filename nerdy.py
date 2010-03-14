import scipy
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
import pylab
import numpy as np

def moneyness(strike, forward, sigma, t):
    return (np.log(strike) - np.log(forward)) / (sigma * np.sqrt(t))

def forward_price(spot, r, t):
    return spot * np.exp(r * t)

def find_implied_forward(synthetic_bids, synthetic_offers):
    best_bid = max(synthetic_bids)
    best_offer = min(synthetic_offers)
    IF = (best_bid + best_offer) / 2
    warning = None
    if max(synthetic_bids) > min(synthetic_offers):
        avg_mid =  np.average((synthetic_bids + synthetic_offers) / 2)
        warning = """\n
                 WARNING: Crossed synthetic market
                 nbbo is %s @ %s
                 bid:   %s, %s
                 offer: %s, %s
                 average: %s
              """ % (best_bid, best_offer,
                    np.average(synthetic_bids), np.std(synthetic_bids),
                    np.average(synthetic_offers), np.std(synthetic_offers),
                    avg_mid)
        print warning
    if warning is not None:
        return avg_mid
    else:
        return IF

def spline_fit(strikes, implied_vols, degree=4):
    '''
    No idea what spline.get_coeffs() is giving me... scrapping this

    Unused for now. Keeping for reference.
    '''
    spline = UnivariateSpline(strikes, implied_vols, w=None, k=degree)
    return spline

def nth_degree_poly(n):
    def f(x, *p):
        return sum([p[i]*x**i for i in range(n)])
    return f

def line(x, p1, p0):
    return (p1*x) + p0

def quadratic(x, p2, p1, p0):
    return (p2*x**2) + (p1*x) + p0

def cubic(x, p3, p2, p1, p0):
    return (p3*x**3) + (p2*x**2) + (p1*x) + p0

def quartic(x, p4, p3, p2, p1, p0):
    return (p4*x**4) + (p3*x**3) + (p2*x**2) + (p1*x) + p0

# TODO: i hate this how can i generate the necessary function?

def polyfit_weighted(strikes, implied_vols, degree=4, w=None):
    if degree == 1:
        general_form = line 
    if degree == 2:
        general_form = quadratic
    if degree == 3:
        general_form = cubic
    else:
        general_form = quartic
    popt, pcov = curve_fit(general_form, strikes, implied_vols, sigma=w)
    return popt

def polyfit_unweighted(strikes, implied_vols, degree=4):
    '''Fit a vol smile'''
    poly_coeffs = scipy.polyfit(strikes, implied_vols, degree)
    return poly_coeffs

def fit_smile(strikes, implied_vols, degree=4, w=None):
    return polyfit_unweighted(strikes, implied_vols, degree=degree)
    # return polyfit_weighted(strikes, implied_vols, degree=4, w=w)

def clip_wings(strikes, implied_vols):
    '''
    Clips wings of strikes and implied vols.

    Removes endpoints and any points adjacent with values
    equal to the relevant endpoint

    This is awful:
        1. np arrays don't have a pop function, and I'm probably using
           np arrays as inputs. converting to list and back
        2. wtf. seriously just repeating code.
        gets the job done for now. refactor later.
    '''
    if implied_vols[0] != implied_vols[1] and implied_vols[-2] != implied_vols[-1]:
        return strikes, implied_vols
    def remove_repeated_from_left(strikes, implied_vols):
        while implied_vols[0] == implied_vols[1] and len(implied_vols) > 2:
            implied_vols.pop(0)
            strikes.pop(0)
    strikes = list(strikes); implied_vols = list(implied_vols)
    for ihatepython in ["fuck", "this"]:
        remove_repeated_from_left(strikes, implied_vols)
        strikes.reverse(); implied_vols.reverse()
    strikes = np.array(strikes); implied_vols = np.array(implied_vols)
    return strikes, implied_vols

def clip_repeated_wings(*data):
    '''Clips numpy vectors on both ends if any vector repeats.

        Inputs: Any number of numpy arrays
        Process: If any array has repeated elements on either end, repeated values
                 and corresponding indexed values of other arrays are popped.
        Returns: Clipped numpy arrays
    '''
    def clip_left(data):
        if len(data[0]) < 2:
            return [[] for d in data]
        repeat_found = any([d[0] == d[1] for d in data])
        if repeat_found:
            [d.pop(0) for d in data]
            return clip_left(data)
        else:
            return data
    def clip_right(data):
        [d.reverse() for d in data]
        data = clip_left(data)
        [d.reverse() for d in data]
        return data
    all_same_length = all([len(d) == len(data[0]) for d in data])
    if not all_same_length:
        msg = "Inputs not all same length"
        raise Exception, msg
    list_data = [list(d) for d in data]
    list_data = clip_left(list_data)
    list_data = clip_right(list_data)
    data = [np.array(d) for d in list_data]
    if len(data) == 1:
    # don't return a list of one nparrays
        data = data[0]
    return data
