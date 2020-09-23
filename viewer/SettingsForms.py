from django import forms


class SettingsForms(forms.Form):
    max_people = forms.CharField(label='number people', max_length=5)
    max_time = forms.CharField(label='max time', max_length=5)
