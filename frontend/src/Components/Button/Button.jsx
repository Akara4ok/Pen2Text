import React from 'react';
import classes from './Button.scss';

class Button extends React.Component {
    constructor(props) {
        super(props);
    }
    render() {
        const { children, className } = this.props;
        return (
            <button className={`${classes.buttonComponent} ${className ?? ''}`}>
                {children}
            </button>
        );
    }
}

export default Button;
