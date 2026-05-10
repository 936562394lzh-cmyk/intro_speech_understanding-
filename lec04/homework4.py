def next_birthday(date, birthdays):
    '''
    Find the next birthday after the given date.

    @param:
    date - a tuple of two integers specifying (month, day)
    birthdays - a dict mapping from date tuples to lists of names, for example,
      birthdays[(1,10)] = list of all people with birthdays on January 10.

    @return:
    birthday - the next day, after given date, on which somebody has a birthday
    list_of_names - list of all people with birthdays on that date
    '''
  
    if not birthdays:
        return (1, 1), [] 

    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    def day_of_year(month, day):
      
        return sum(month_days[:month-1]) + day

    given_day = day_of_year(date[0], date[1])

    next_day = None      
    next_date = None    
    next_names = None    

    min_day = None      
    min_date = None      
    min_names = None    

    for bdate, names in birthdays.items():
        bday = day_of_year(bdate[0], bdate[1])

        if min_day is None or bday < min_day:
            min_day = bday
            min_date = bdate
            min_names = names

        if bday > given_day:
            if next_day is None or bday < next_day:
                next_day = bday
                next_date = bdate
                next_names = names

    if next_date is not None:
        return next_date, next_names
    else:
        return min_date, min_names
