class Data_Config_Count():
    def __init__(self, config_zone=False, config_intesection_zone=False,
                 config_remove_area=False, config_num_people_total=False,
                 config_last_reset=False):

        self.config_zone = config_zone
        self.zone = {}

        self.config_intesection_zone = config_intesection_zone
        self.intesection_zone = {}

        self.config_remove_area = config_remove_area
        self.remove_area = {}

        self.config_num_people_total = config_num_people_total
        self.num_people_total = 0

        self.config_last_reset = config_last_reset
        self.last_reset = None

    def set_num_people(self, num):
        if self.config_num_people:
            self.num_people = num
            return True
        return False
