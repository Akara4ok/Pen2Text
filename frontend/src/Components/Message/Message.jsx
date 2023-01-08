import React from 'react';
import classes from './Message.scss';
import Button from '@Components/Button/Button'

class Message extends React.Component {
    constructor(props) {
        super(props);
    }
    render() {
        const { children } = this.props;
        return (
            <div className={classes.message}>
                <div className={classes.text}>{children}</div>
                <Button className={classes.button} onClick={this.props.onClose}>
                    Ok
                </Button>
            </div>
        );
    }
}

export default Message;
