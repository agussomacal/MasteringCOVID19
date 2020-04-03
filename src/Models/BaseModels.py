import copy


class ModelBase:
    def equation(self, **kwargs):
        pass

    def get_equation_for_odeint(self):
        eqparam_names = self.get_modelvar_names(self, without_time=False)

        def equation_for_odeint(x, t):
            values4eq = list(x) + [t]
            return self.equation(**{var_name: value4eq for var_name, value4eq in zip(eqparam_names, values4eq)})

        return equation_for_odeint

    def get_func_for_optimization(self):
        def obj_func(**params):
            other = copy.deepcopy(self)
            for param, value in params.items():
                other.__setattr__(name=param, value=value)
            return other.equation

        return obj_func

    @staticmethod
    def get_modelparam_names(class_model):
        """
        model parameters
        :param class_model:
        :return:
        """
        return class_model.__init__.__code__.co_varnames[1:class_model.__init__.__code__.co_argcount]

    @staticmethod
    def get_modelvar_names(class_model, without_time=True):
        """
        model variables
        :param class_model:
        :return:
        """
        output_num_vars = (class_model.equation.__code__.co_argcount - 2)  # -self -t
        res = class_model.equation.__code__.co_varnames[1:]
        return res[:output_num_vars] if without_time else res[:output_num_vars + 1]
