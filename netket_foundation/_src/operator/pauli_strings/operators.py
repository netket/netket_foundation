from netket_foundation._src.operator.pauli_strings.jax import PauliStringsJax


def sigmax(hi, i, dtype=None, mode="index"):
    """Pauli X operator acting on site `i` of the Hilbert space `hi`."""
    pstring = list("I" * hi.size)  # Convert string to list of characters
    pstring[i] = "X"  # Modify the desired index
    pstring = "".join(pstring)  # Join the list back into a string
    return PauliStringsJax(hi, [pstring], _mode=mode, dtype=dtype)


def sigmay(hi, i, dtype=None, mode="index"):
    """Pauli Y operator acting on site `i` of the Hilbert space `hi`."""
    pstring = list("I" * hi.size)  # Convert string to list of characters
    pstring[i] = "Y"  # Modify the desired index
    pstring = "".join(pstring)  # Join the list back into a string
    return PauliStringsJax(hi, [pstring], _mode=mode, dtype=dtype)


def sigmaz(hi, i, dtype=None, mode="index"):
    """Pauli Z operator acting on site `i` of the Hilbert space `hi`."""
    pstring = list("I" * hi.size)  # Convert string to list of characters
    pstring[i] = "Z"  # Modify the desired index
    pstring = "".join(pstring)  # Join the list back into a string
    return PauliStringsJax(hi, [pstring], _mode=mode, dtype=dtype)
