

#############################################################################

def find_fibre_table(hdulist):
    """Returns the extension number for FIBRES_IFU, MORE.FIBRES_IFU FIBRES or MORE.FIBRES,
    whichever is found. Modified from SAMI versiuon that only uses FIBRES_IFU.
    Raises KeyError if neither is found."""

    extno = None
    try:
        extno = hdulist.index_of('FIBRES')
    except KeyError:
        pass

    if extno is None:
        try:
            extno = hdulist.index_of('MORE.FIBRES')
        except KeyError:
            pass

    if extno is None:            
        try:
            extno = hdulist.index_of('FIBRES_IFU')
        except KeyError:
            pass
        
    if extno is None:
        try:
            extno = hdulist.index_of('MORE.FIBRES_IFU')
        except KeyError:
            raise KeyError("Extensions 'FIBRES_IFU' and "
                           "'MORE.FIBRES_IFU' both not found")
    return extno
