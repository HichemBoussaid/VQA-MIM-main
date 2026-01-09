class Objects:
    def __init__(self, obj, features, exclude, parent=None):
        self.plural_string = None
        self.features = features
        self.exclude = []
        self.count = 0
        self.success = True
        self.string = 'ERROR'
        self.type = 'None'
        self.parent = parent
        # self.object_list = []
        self.PROBA_ATTRIBUTE = .1
        self.ATTRIBUTES = {'size': ['small', 'medium', 'large'], 'shape': ['square', 'rectangular', 'circular']}
        self.constructionObjects(obj, exclude)

    def constructionObjects(self, obj, exclude):
        self.string = obj
        self.count = len(self.features)
        self.plural(True)
        self.exclude = exclude

    def plural(self, plural):
        if plural and (len(self.string.split(" ")) == 1 or 'plantation' in self.string):
            if self.string[-1] == 'y' and not self.string[-2] in ['u', 'a', 'i', 'e', 'o']:
                self.plural_string = self.string[:-1] + "ies"
            elif self.string[-1] in ['s', 'x', 'z'] or self.string[-2:] in ['ss', 'sh', 'ch']:
                self.plural_string = self.string + 'es'
            else:
                self.plural_string = self.string + 's'
        else:
            self.plural_string = self.string
