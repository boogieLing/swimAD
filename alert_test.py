from associate.alert import risk_alert
from entries.track_entry import AssociateResult


if __name__ == "__main__":
    message = []
    message.append(AssociateResult([1,2,3,4], 1.0, 2.0, "1111"))
    risk_alert(message)