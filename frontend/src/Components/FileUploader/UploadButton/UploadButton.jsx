import React from 'react';
import classes from './UploadButton.scss';

class UploadButton extends React.Component {
    constructor(props) {
        super(props);
    }
    render() {
        return (
            <div className={classes.wrapper}>
                <button>upload</button>
            </div>
        );
    }
}

export default UploadButton;
